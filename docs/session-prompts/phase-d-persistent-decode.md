# Phase D · D2 — plan-once persistent decode (design)

**Status:** DESIGN ONLY (2026-06-30). No behaviour code in this PR. This document is the
PR-by-PR plan for D2, grounded against the as-built code at worktree
`b0-3-backend-contract` = `origin/main` HEAD `4585b194` (D1 landed). It refines §0 step 3
("persistent decode wiring") of
[`symbolic-extents-and-persistent-decode.md`](symbolic-extents-and-persistent-decode.md).

> **Scope:** single-session D2 only. Concurrent `(NodeId, SessionId)` keying is D3; the API
> below is shaped so D3 slots in without rework (see §8). PhiModel decode stays on the sliced
> per-token-rebuild path (D4).

---

## 0. The win, restated

D1 (`4585b194`) made the LlamaModel decode-step graph **structurally + shape-identical every
token**: fixed-capacity K/V `[1, Hkv, max_seq_len, D]`, KV write via `write_slice_dyn` at
`DynScalar::Sym(SymId(0))`, attention over the full capacity with a fixed
`[1, 1, seq, max_seq_len]` causal mask, all resolved through a per-pass `SymEnv` bound to
`cached_len`. The byte-exact gate `forward_with_kv_context_decode_matches_non_cached_forward`
passes.

But the graph is still **rebuilt fresh and re-optimized every token** — see
`forward_with_kv_context_impl` (`fuel-core/src/lazy.rs:5212`): it constructs a brand-new
`LazyTensor` graph (fresh `NodeId`s for token-ids, RoPE cos/sin, per-layer mask, per-layer KV
placeholders), realizes via `ctx.realize_one_as_with_env`, then drops the graph. Each realize
runs `optimize_graph` (the placement DP + cost composer + Judge + residency/layout passes) from
scratch.

**D2 = build the decode-step graph ONCE, optimize it ONCE, then per token only re-bind the
data + the `SymEnv` and re-realize the SAME graph with the bridge SKIPPING the optimize.** The
~1.8×/token comes from not re-planning. Must stay byte-exact vs. the D1 per-token-replan path.

---

## 1. The realize pipeline today — what mutates the graph (grounded)

Per-token, `ctx.realize_one_as_with_env(graph, root, env)` →
`pipelined_bridge::realize_one_as_with_initial_env` → `realize_one_as_with_initial_reporting`
(`fuel-core/src/pipelined_bridge.rs:250,276`). That fn does, in order:

1. **`prepare(graph, [target], device, initial)`** (`pipelined_bridge.rs:731` →
   `prepare_split:751`). **MUTATES the graph** (write lock):
   - **splices an `Op::Copy { target: Cpu }` at every realize root** (D2H), returning the Copy
     `NodeId` as the `effective_target` (`prepare_split:783-809`);
   - builds the `StorageCache` of all reachable `Op::Const` via `build_const_cache:866` on top
     of `initial` (the cloned `ctx.persistent`). Const slots already present in `initial` (the
     KV placeholders) are skipped; the data Consts (token-ids, RoPE, mask) are uploaded from
     `graph.storage_for(id)`.
2. **`dispatch_with_plan_retry(...)`** (`pipelined_bridge.rs:468`), looping on
   `TopologyChanged`:
   - **`build_optimized_graph(graph, [cpu_target], pinned_loc, &cache)`** (`:416`) →
     **`optimize_graph(&mut g, roots, &bindings, &options)`** (`fuel-dispatch/src/optimize.rs:194`).
     This is the expensive step and **MUTATES the graph in place**: it runs `compile_plan` (the
     placement DP / cost composer / per-node ranker / Judge oracle), then **stamps
     `target_backend`** on every kernel node (`stamp_plan_backends:294`), **inserts residency
     `Op::Copy`** (`insert_residency_copies:339`), and **inserts layout `Op::Contiguize`**
     (`insert_layout_fixups`, optimize.rs:269). Returns the tiny `OptimizedGraph` **view**.
   - builds the runtime `(selector, lookup)` via `production_selector_for(device)` (`:507`);
   - **`PipelinedExecutor::realize_with_optimized_picking_env(graph, cpu_target, cache, &optimized,
     selector, lookup, sym_env)`** (`fuel-dispatch/src/pipelined.rs:631`).
3. Inside the executor, **`realize_inner`** (`pipelined.rs:685`) ALSO **mutates the graph** (write
   lock): `insert_safety_copies(&mut g, roots)` at the top (`:699`), then on a read lock builds
   the per-call `CompilerWork` dispatch order + seeds `layout_cache`, spawns the compiler thread,
   and dispatches. `compile_plan` is **NOT** re-run here — kernel resolution is via the binding
   registry over the (already-stamped) graph.

### The three in-place graph mutations that make naive re-realize unsafe

| Mutation | Where | Re-run on held graph ⇒ |
| --- | --- | --- |
| D2H `Op::Copy` splice at roots | `prepare_split:801` | **double-splices** a 2nd Copy on the 1st Copy (extra D2H + new root) |
| Placement stamps + residency `Op::Copy` + layout `Op::Contiguize` | `optimize_graph` (`stamp`/`insert_residency_copies`/`insert_layout_fixups`) | **double-inserts** copies/contiguizes → corruption; this is the explicit hazard the prompt names |
| Safety-copy `Op::Copy` for destructive-op readers | `realize_inner` → `insert_safety_copies:1596` | see analysis below |

**`insert_safety_copies` analysis (fuel-graph/src/opt.rs:1596):** it finds destructive ops
(`Op::WriteSlice` is destructive on its dest input) whose `target` has *conflicting readers*
(readers not provably ordered before the write), inserts an `Op::Copy` of the target, and
`rewrite_input`s those readers onto the copy. After the **first** realize rewrites the readers,
on the **second** realize `graph.node(reader).inputs.contains(&target)` is false for those
readers → the same conflict does not re-fire. So it is a **fixpoint after the first pass** for a
structurally-stable graph: not corrupting on re-run, but it still takes a write lock + re-derives
ordering every call (a per-token cost we can also elide — see PR D2a note). For the decode graph,
whether it fires at all depends on the WriteSlice→attention edge ordering; **must be measured**
(open question Q4).

### Critical cacheability finding — `OptimizedGraph` bakes NO Const data

`OptimizedGraph` (`fuel-dispatch/src/optimize.rs:101-150`) holds **only**
`{ roots: Vec<NodeId>, generation: u64 }`. Its `runs()`/`dispatch_order()`/`branch_count()` are
**derived on demand from the passed-in `&Graph`** (`extract_runs_multi` / `lower_runs_arm0`). It
references **nothing** about Const data, storage, or the `SymEnv`. The durable optimization output
lives **in the mutated graph** (the `target_backend` stamps + inserted `Op::Copy`/`Op::Contiguize`
+ `Op::Branch` arms), NOT in the view. **Consequence:** caching the `OptimizedGraph` across tokens
is sound **iff the graph structure is identical across tokens** — which is exactly D1's guarantee
— because the cached view is just `{roots, generation}` over a graph whose only per-token change
is Const *bytes* and the `SymEnv`. The `generation` is the `SystemTopology` generation; reusing it
keeps the executor's `TopologyChanged` chunk-boundary check honest (if a backend hot-plugs, the
live generation diverges and the executor surfaces `TopologyChanged` → we rebuild; see §5
invalidation).

### Selector/lookup reusability

`(selector, lookup) = production_selector_for(device)` (`pipelined_bridge.rs:507`) is
**Device + Judge-derived**, not graph- or data-derived. For a branchless decode graph (D1 emits
no `Op::Branch`), `streaming_pick_for` returns `None` and the executor takes the arm-0
`realize_with_optimized_env` path regardless. So the selector is effectively a no-op for decode
today; reusing or rebuilding it per token is immaterial. Design: rebuild it cheaply per token
(it's a couple of hashmap lookups) OR cache it on the session — either is fine; **prefer caching
it on the session** to keep the hot path allocation-free. (Open question Q3: confirm
`production_selector_for` has no per-realize state.)

---

## 2. What varies per token (and how each becomes a stable re-bindable Const)

Today `forward_with_kv_context_impl` mints fresh `NodeId`s every token. For plan-once we build
**once** with STABLE `NodeId`s and re-bind the DATA each token. Per-token-varying inputs:

| Input | Today's builder | Bytes change per token? | Re-bind plan |
| --- | --- | --- | --- |
| **token-ids** | `embed.const_u32_like(tokens, [seq])` (`lazy.rs:5264`) | yes (the new token id) | stable Const `NodeId`; per token build a fresh `Arc<RwLock<Storage>>` of the `[seq]` u32 and `ctx.insert(token_id, arc)` |
| **RoPE cos/sin** | `h.rope_tables_const(base, cached_len, seq, head_dim)` (`lazy.rs:5270`) | yes (position = `cached_len`) | two stable Const `NodeId`s; per token recompute the `[seq, head_dim]` cos/sin host tables from `cached_len` and `ctx.insert` each |
| **causal mask** | `x.const_f32_like(mask_data, [1,1,seq,max_seq_len])` (`lazy.rs:5126-5136`) | yes (−∞ tail shifts with `cached_len`) | one stable Const `NodeId` per layer (built per layer today — see note); per token recompute the `[1,1,seq,max_seq_len]` mask and `ctx.insert` |
| **KV cache K/V** | `h.const_placeholder_like(cache_shape, dtype)` (`lazy.rs:5304-5305`) + `ctx.insert(k_id, k_arc)` | Arc is stable; bytes mutate in place via WriteSlice | **already** the stable pattern; the `Arc` persists in the cache, the placeholder `NodeId` becomes stable |
| **`cached_len` symbol** | `SymEnv::new(); bind(SymId(0), cached_len)` (`lazy.rs:5358`) | yes | re-bind a fresh `SymEnv` per token (already per-pass) |

**The re-bind mechanism already exists** and is exactly what the KV path uses:
`const_placeholder_like` pushes a Const node WITHOUT seeding `graph.storage_map`, and
`InferenceContext::insert(node_id, arc)` makes the realize-time `StorageCache` (cloned from
`ctx.persistent`) carry that node's bytes, short-circuiting the `build_const_cache` upload walk
(see the `forward_with_kv_context` doc-comment, `lazy.rs:5168-5178`, and `build_const_cache`'s
"persistent slots take precedence", `pipelined_bridge.rs:877`). D2 generalizes this from "only the
KV Arcs" to "token-ids, RoPE, mask, and KV are ALL `const_placeholder_like` + per-token
`ctx.insert`".

### The mask + RoPE multiplicity wrinkle

- The **mask** is currently built *inside* `apply_layer_with_kv_writes` (`lazy.rs:5126`), so there
  is one mask Const **per layer** — but they are byte-identical across layers (mask depends only on
  `cached_len`, `seq`, `max_seq_len`). For the held graph we should **hoist the mask to a single
  Const built once in `forward_with_kv_context_impl`** and pass it into each layer (like RoPE
  tables already are), so there is ONE stable mask `NodeId` to re-bind per token instead of
  `n_layers`. (This is also a graph-size reduction — CSE would dedup them anyway, but building one
  is cleaner and halves the re-bind count.) **This is a D2b refactor, byte-exact by construction**
  (same mask data, fewer nodes).
- **RoPE** cos/sin are already built once and shared (`lazy.rs:5270`) → two stable `NodeId`s.

### Why the bytes can be rebound under a stable NodeId without re-uploading on CPU

On CPU, `build_const_cache` short-circuits: the persistent map entry (the `ctx.insert`ed Arc) is
used directly (`pipelined_bridge.rs:877` "persistent slots take precedence"). On non-CPU, the Arc
in `ctx.persistent` is already a **device-resident** storage (the KV path proves this); for the
re-bound data Consts (token/RoPE/mask) we must insert **device-resident** Arcs too, or accept a
per-token H2D. Two options (Q1):
- **(a)** keep token/RoPE/mask as ordinary graph `Op::Const`s with `graph.storage_map` bytes that
  we *mutate in place* each token (write new bytes into the existing storage Arc). Then
  `build_const_cache` re-uploads them H2D each token — small tensors, but a per-token H2D + a
  per-token `build_const_cache` walk (which we want to skip).
- **(b)** make them `const_placeholder_like` + `ctx.insert` of a **device-resident** Arc we write
  host bytes into each token (one small H2D per token via the same path `KvCache::with_capacity`
  uses, or a host→device write helper). This keeps them out of the `build_const_cache` walk.

**Recommend (b)**: it is the KV pattern, it keeps the const-cache walk skippable, and the H2D for
token([seq]≈1) + RoPE([seq,D]) + mask([seq,max_seq]) is tiny. The mask is the largest
(`seq*max_seq_len` f32); for seq==1 decode that is `max_seq_len` floats — negligible. **Decision
needed before D2b (Q1).** On CPU-only (the born-red test bed) this is a non-issue.

**Q1 verified-feasible:** the per-token device-overwrite helper EXISTS —
`CudaStorageBytes::write_from_host(&[u8])` (`fuel-cuda-backend/src/byte_storage.rs:294`, asserts
`src.len() == self.len_bytes`) and `VulkanBackend::write_bytes` (`fuel-vulkan-backend/src/lib.rs`).
On CPU the storage is host bytes, overwrite is a slice copy. So option (b) needs a small
`DecodeSession` helper that, per token, locks each data-Const's persistent Arc and writes fresh
host bytes into it via the backend-appropriate path (the same family `build_const_cache`'s non-CPU
arm uses). The Arc identity + NodeId stay stable; only the bytes change. Note: this writes
**outside** the graph (a direct storage mutation, like the KV WriteSlice mutates in place), which
is consistent with how KV bytes already evolve across tokens.

---

## 3. Where the held state lives — `DecodeSession`

Add a held-graph struct rather than overloading `InferenceContext` (which is the per-session
storage map and should stay storage-only). Proposed location:
`fuel-core/src/inference_context.rs` (sibling to `InferenceContext`) or a new
`fuel-core/src/decode_session.rs`. Shape:

```rust
/// Plan-once persistent decode state for one LlamaModel + one KvCache
/// capacity/dtype. Built on the first seq==1 decode token; reused (graph
/// + plan held, only data + SymEnv re-bound) for every subsequent token.
pub struct DecodeSession {
    /// The held decode-step graph, ALREADY optimized in place (stamps +
    /// residency/layout copies + D2H root splice baked in). Structure is
    /// stable across tokens (D1 guarantee).
    graph: Arc<RwLock<Graph>>,
    /// The cached optimize view from the first realize. Holds only
    /// {roots, generation}; valid while `graph` structure + topology
    /// generation are unchanged (§1 finding).
    optimized: OptimizedGraph,
    /// The realize root the executor was asked for — the D2H Op::Copy
    /// NodeId that `prepare` spliced (NOT the logits node itself).
    effective_target: NodeId,
    /// The logits node (pre-D2H-splice) — `effective_target`'s input.
    logits_node: NodeId,
    /// Stable re-bindable data-Const NodeIds.
    token_ids_node: NodeId,
    rope_cos_node: NodeId,
    rope_sin_node: NodeId,
    mask_node: NodeId,
    kv_nodes: Vec<(NodeId, NodeId)>, // (k_const, v_const) per layer
    /// The symbol the SymEnv binds each token.
    cached_len_sym: SymId,
    /// Validity key — rebuild if any of these change vs. the live cache/model.
    max_seq_len: usize,
    n_layers: usize,
    cache_dtype: DType,
    // (selector, lookup) optionally cached here too — Q3.
}
```

`DecodeSession` is held by the **caller of the generate loop** (the harness that owns the
`LlamaModel` + `KvCache` + `InferenceContext`), constructed lazily on the first decode token. It is
NOT held by `LlamaModel` (the model is immutable weights; a session is per-generation state) — but
a thin `Option<DecodeSession>` field on a generation-loop struct, or returned from a
`begin_decode(...)` builder, both work. **Decision: hold it beside `InferenceContext` in whatever
struct the generate loop owns** (today the loop is open-coded in tests + callers; D2c introduces
the held entry point — see §6).

---

## 4. The optimize-skip seam (the heart of D2)

We need a bridge realize variant that, given an **already-prepared, already-optimized** graph + a
cached `OptimizedGraph`, goes straight to the executor and skips BOTH `prepare` (no re-splice) AND
`build_optimized_graph` (no re-plan). Proposed signature in `pipelined_bridge.rs`:

```rust
/// Plan-once realize: the graph has ALREADY been `prepare`d (D2H Op::Copy
/// spliced) and `optimize_graph`'d (stamps + residency/layout baked) on a
/// prior call; `optimized` is the cached view from that call. Re-binds
/// only `cache` (the per-token StorageCache, incl. the freshly re-bound
/// data Consts) + `sym_env`, then dispatches. SKIPS prepare + optimize.
///
/// `effective_target` is the D2H Op::Copy NodeId `prepare` returned the
/// first time (stable across tokens). `report_node` is the logits node.
pub fn realize_one_prebuilt_env<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    effective_target: NodeId,
    report_node: NodeId,
    optimized: &OptimizedGraph,
    device: &Device,
    cache: StorageCache,
    sym_env: &SymEnv,
) -> Result<Vec<T>> {
    // NO prepare(), NO build_optimized_graph(). Straight to:
    let (selector, lookup) = production_selector_for(device).map(...);  // or cached
    let (storage, _src) = /* the TopologyChanged-retry-wrapped */
        PipelinedExecutor::realize_with_optimized_picking_env(
            graph.clone(), effective_target, cache, optimized,
            selector, lookup, sym_env.clone(),
        )?;
    Ok(extract_cpu_bytes_typed::<T>(&storage)?)
}
```

The `TopologyChanged` retry is the wrinkle: today retry re-runs `build_optimized_graph` (which is
what makes a stale-topology rebuild correct). On the prebuilt path, a `TopologyChanged` means the
cached `optimized.generation` is stale and the stamps may be wrong → we must **fall back to a full
rebuild** (re-`prepare` is wrong since the graph is already spliced, so the fallback re-optimizes
the *already-prepared* graph against the fresh topology, or — cleaner — invalidates the whole
`DecodeSession` and rebuilds it from scratch next token). **Recommend: on `TopologyChanged`,
invalidate the `DecodeSession` (drop it) and rebuild the graph fresh** — topology changes mid-decode
are rare (a device hot-plug) and a one-token rebuild is acceptable. (Q5.)

`StorageCache` per token: clone `ctx.persistent` (now containing the re-bound token/RoPE/mask/KV
Arcs) — exactly `InferenceContext::cloned_persistent()`. Because all data Consts are persistent
(option (b) §2), `build_const_cache` is not needed at all on the prebuilt path → the const-cache
walk is also skipped (a secondary win).

A new `InferenceContext` method wraps it:
```rust
pub fn realize_prebuilt_as_with_env<T>(&self, session: &DecodeSession, sym_env: &SymEnv) -> Result<Vec<T>>
```
that calls `realize_one_prebuilt_env` with `session.graph`, `session.effective_target`,
`session.logits_node`, `&session.optimized`, `self.device`, `self.cloned_persistent()`, `sym_env`.

### Does `insert_safety_copies` need to be skipped too?

`realize_with_optimized_picking_env` → `realize_inner` runs `insert_safety_copies` every call. Per
§1 it is a fixpoint after the first realize (no corruption), but it costs a write-lock + ordering
derivation per token. **Option:** add a `realize_inner` variant (or a flag on `OrderSource`) that
skips `insert_safety_copies` when the caller asserts the graph is already safety-copied (the
prebuilt path can assert this, since the first full realize ran it). This is a **secondary
optimization** — measure first (Q4); if `insert_safety_copies` doesn't fire on the decode graph at
all (no conflicting readers of the WriteSlice target), it is already cheap-ish and we can defer the
skip. **Keep D2a minimal: skip prepare + optimize only; leave safety-copy skipping to a follow-up
if the profile shows it.**

---

## 5. Rebuild / invalidation policy

Build the `DecodeSession` lazily and hold it while valid. Invalidate (drop + rebuild next token)
when any of:

- **first decode token** — no session yet → build it (full `prepare` + `optimize` path, cache the
  `OptimizedGraph` + node ids).
- **`max_seq_len` / `n_layers` / `cache_dtype` change** vs. the live cache — the held graph's
  shapes are keyed to these (the capacity axis, the per-layer KV nodes, the const dtypes). A change
  means a different model/cache → rebuild.
- **`seq != 1`** (prefill, or a multi-token verification step) — the held graph is shape-keyed to
  `seq` (token-ids `[seq]`, mask `[1,1,seq,max_seq_len]`, logits slice). A different `seq` is a
  different graph. **See §7: prefill keeps the rebuild path; the held graph is the seq==1 decode
  graph only.** (A future refinement could hold a small set of graphs keyed by `seq`, but D2 holds
  exactly the seq==1 one.)
- **`TopologyChanged`** at dispatch (a backend hot-plug) — invalidate + rebuild (§4, Q5).
- **cache cleared / new generation** — the caller calls `session = None` (or `DecodeSession::reset`)
  between independent generations. The KV Arcs differ per generation (a fresh `KvCache`), so the
  held graph's KV placeholder bindings are stale; rebuild. (If the same `KvCache` object is reused
  across `clear()` + a new prompt, the Arcs persist and only `cached_len` resets — then the session
  COULD be reused; but be conservative for D2 and rebuild on `clear`. Q6.)

The validity check is a cheap field comparison at the top of the decode entry point.

---

## 6. Generate-loop integration

Today there is no held generate loop in the model API — `forward_with_kv_context` is called
per-token by callers/tests. D2c introduces a held decode entry point. Two shapes (pick one — Q2):

- **(A) A `decode_step` method** on a session-owning struct:
  `fn decode_step(&mut self, token: u32, cache, ctx) -> Result<Vec<f32>>` that (1) validates/builds
  the `DecodeSession`, (2) re-binds token/RoPE/mask data Consts via `ctx.insert`, (3) binds the
  `SymEnv`, (4) calls `ctx.realize_prebuilt_as_with_env`, (5) bumps `cache.cached_len` + versions.
- **(B) Keep `forward_with_kv_context` as the entry**, give it an optional
  `&mut Option<DecodeSession>` param (or a sibling `forward_with_kv_context_persistent`) so the
  byte-exact gate can compare the two paths directly.

**Recommend (B) as a sibling** `forward_with_kv_context_persistent(tokens, cache, ctx, session:
&mut Option<DecodeSession>)`: it lets the born-red test call BOTH the D1 path
(`forward_with_kv_context`) and the D2 path on the same inputs and assert byte-equality + the
optimize-skip, with minimal surface churn. The internal split:

```
forward_with_kv_context_persistent(tokens, cache, ctx, session):
    if seq != 1 or session invalid:                 # §5
        # full rebuild: build graph with STABLE node ids, prepare+optimize ONCE,
        # cache OptimizedGraph + node ids into *session; realize via the
        # existing dispatch_with_plan_retry path; record optimize_ran = true.
    else:
        # re-bind token/RoPE/mask data into ctx.persistent under the held node ids;
        # bind SymEnv(cached_len); realize via realize_prebuilt_env; optimize_ran = false.
    bump cache.cached_len + versions
```

The first call (seq==1, no session) builds + optimizes once; subsequent seq==1 calls skip optimize.

---

## 7. Prefill vs. decode interaction

- **Prefill** (`seq > 1`, the initial prompt) and **spec-decode verification**
  (`forward_with_kv_context_all_positions`, multi-token) keep the **rebuild path** unchanged —
  they run once (prefill) or rarely (verification) and are shape-distinct from the seq==1 decode
  graph. No plan-once for them in D2.
- **Decode** (`seq == 1`, the autoregressive loop) is the only path that gets the held graph. This
  is where the 1.8× lives (thousands of identical-shape tokens).
- The first decode token after prefill is the session **build** (full optimize); every token after
  is the **reuse** (skip optimize). So a 200-token generation pays the optimize cost ~once (decode
  build) + once (prefill), and reuses for ~199 tokens.

This matches §0's "build the decode-step graph once" exactly: the decode graph (seq==1) is the held
artifact.

---

## 8. D3 forward-compat (design the API, don't build it)

D3 = concurrent sessions keyed by `(NodeId, SessionId)`. The D2 API anticipates it:
- `DecodeSession` is already a per-session object (not a global / not on `LlamaModel`), so N
  concurrent decodes hold N `DecodeSession`s + N `InferenceContext`s + N `KvCache`s today.
- The shared-immutable graph + per-session `SymEnv` + per-session `StorageCache` is the
  spec's `[8]`/`[9]` model: the graph carries `SymId`s; bindings are per-pass input. For D3 the
  **graph itself** may be shared across sessions (same model, same capacity) — then the per-session
  data Consts must be keyed by `(NodeId, SessionId)` in the storage map rather than re-`insert`ed
  into one `ctx`. The `realize_one_prebuilt_env` signature already takes the `StorageCache`
  explicitly, so a D3 caller supplies a session-keyed cache without changing the bridge seam.
- **Do not** key anything on global mutable state in D2; everything threads through
  `DecodeSession` + `InferenceContext` so D3 is additive.

---

## 9. PR breakdown (each born-red first)

### D2a — the optimize-skip bridge seam
- **Add** `pipelined_bridge::realize_one_prebuilt_env` (skips `prepare` + `build_optimized_graph`;
  dispatches via `realize_with_optimized_picking_env`; `TopologyChanged` → typed error for the
  caller to handle by invalidation) + `InferenceContext::realize_prebuilt_as_with_env`.
- **Add** an instrumentation hook to count `optimize_graph` invocations. Cleanest:
  an `AtomicU64` `OPTIMIZE_CALLS` counter bumped at the top of `build_optimized_graph`
  (`pipelined_bridge.rs:416`), with a test-only reader. (Alternative: a counter inside
  `optimize_graph` in fuel-dispatch — but the bridge counter is closer to the seam and avoids
  touching fuel-dispatch.)
- **Born-red test (fuel-core, CPU):** build a small graph manually (or a 1-2 layer toy model),
  realize it once via the full path (records `prepare` + 1 optimize, captures the `OptimizedGraph`
  + effective target), then realize the SAME graph via `realize_one_prebuilt_env` with a different
  `SymEnv`/Const binding and assert: (1) `OPTIMIZE_CALLS` did NOT increment on the 2nd realize,
  (2) the result is byte-identical to a full-path realize of the same inputs, (3) no new
  `Op::Copy`/`Op::Contiguize` nodes were added on the 2nd realize (graph `len()` unchanged).
  Red before the seam exists (the method is absent / counter not wired); green after.

#### D2a — landed (2026-06-30, uncommitted on `b0-3-backend-contract`)

**Mechanism as built** (all in `fuel-core/src/pipelined_bridge.rs` + `inference_context.rs`, CPU-only,
backend-agnostic seam; no `fuel-dispatch` change):

- **`realize_one_prebuilt_env<T>(graph, effective_target, optimized, device, cache, sym_env)`** —
  the skip seam. Goes STRAIGHT to `PipelinedExecutor::realize_with_optimized_picking_env`, bypassing
  BOTH `prepare` (no D2H `Op::Copy` re-splice, no `build_const_cache` walk) AND
  `build_optimized_graph`/`optimize_graph` (no placement re-plan, no double-inserted residency
  `Op::Copy`/layout `Op::Contiguize`). It reuses the cached `OptimizedGraph` view directly — sound
  because the view holds only `{roots, generation}` and the durable optimize output lives in the
  already-mutated (stamped + copy-stitched) graph, valid while structure + topology generation are
  stable (D1 guarantee). `(selector, lookup)` are rebuilt per call via `production_selector_for`
  (per-realize-stateless, no-op for the branchless decode graph — arm-0 lowering). `TopologyChanged`
  is surfaced as its **typed error, NOT retried** (a topology shift means the cached generation is
  stale → the caller invalidates + rebuilds the session, per §4/§5).
- **`prebuild_optimized_env<T>(...) -> (effective_target, OptimizedGraph, Vec<T>)`** — the
  return-path a first normal realize takes so D2b can build the `DecodeSession`. It runs `prepare` +
  optimize + dispatch ONCE (byte-identical value to `realize_one_as_with_initial_env`) and
  additionally surfaces the spliced D2H `Op::Copy` root (`effective_target`) + the cached
  `OptimizedGraph` + the first token's bytes. Backed by a `dispatch_with_plan_retry_capturing`
  sibling of `dispatch_with_plan_retry` that keeps the FINAL successful `OptimizedGraph` instead of
  dropping it.
- **`InferenceContext::prebuild_optimized_as_with_env` + `realize_prebuilt_as_with_env`** — the
  context wrappers (thread `self.device` + `self.cloned_persistent()`), the entry D2b calls.
- **Instrumentation:** `OPTIMIZE_CALLS: AtomicUsize` (process-global, `Relaxed`, mirrors the B1
  in-flight-counter idiom) bumped at the top of `build_optimized_graph`; `pub fn optimize_calls()`
  reads it. **Deviation from the spec (necessary):** a *process-global* counter cannot give the test
  an isolated delta because ~1000 other suite tests bump it concurrently (observed: the naive
  absolute-count assertion failed `1924 -> 1927` in the full suite). So a **thread-local mirror**
  `OPTIMIZE_CALLS_TL` is bumped in the same spot (`build_optimized_graph` always runs on the realize
  CALLER's thread — the optimize takes the graph write-lock synchronously BEFORE the executor spawns
  its compiler thread), with `pub fn optimize_calls_thread_local()`; the test asserts on that
  isolated per-thread delta. The process-global counter stays as the coarse telemetry surface the
  prompt asked for. `AtomicUsize` (not `AtomicU64`) matches the existing `tensor.rs` counter idiom.
- Existing realize entries left byte-identical (`realize_one_as_with_initial_reporting` etc.
  unchanged; the counter bump is additive).

**Test:** `pipelined_bridge::tests::d2a_prebuilt_realize_skips_optimize_and_does_not_grow_graph`.
First realize via `prebuild_optimized_env`; 2nd via `realize_one_prebuilt_env` (fresh per-call cache +
empty `SymEnv`). Asserts (a) `optimize_calls_thread_local()` unchanged across the prebuilt realize,
(b) 2nd result `==` 1st **exactly** (not epsilon), (c) graph `len()` unchanged. Includes a **control**
that re-realizes the same graph through the full path and asserts the optimizer DOES bump + the graph
DOES grow (the double-splice hazard the seam avoids). **Observed red** (probe routing the 2nd realize
through the full path failed assertion (a) `1 -> 2`), **then green** after wiring the seam, and green
in the full suite (**1284 passed**, was 1283 at HEAD; 0 failed, 10 ignored).

### D2b — held-graph build + data re-bind (LlamaModel)
- **Hoist the mask** out of `apply_layer_with_kv_writes` to a single Const in
  `forward_with_kv_context_impl` (byte-exact refactor; born-red is the existing decode gate still
  passing).
- **Add** `DecodeSession` (struct + lazy build + validity check, §3/§5).
- **Add** `forward_with_kv_context_persistent(tokens, cache, ctx, session)` (§6 shape B): seq==1 +
  valid → re-bind data Consts + realize prebuilt; else full rebuild that captures the session.
- Decide + implement the device-resident data-Const binding (§2 option (b)) — on CPU it is the
  trivial path; gate the H2D detail behind the existing `ctx.insert` of a device Arc.
- **Born-red test (fuel-core, CPU):** drive `forward_with_kv_context_persistent` for ≥3 decode
  tokens with a held `session`, asserting (1) `OPTIMIZE_CALLS` increments exactly ONCE across the
  ≥3 tokens (the build), not per token; (2) each token's logits are byte-identical to the D1
  `forward_with_kv_context` path on the same prefix; (3) the held graph's node `len()` is stable
  from token 2 onward (no per-token node growth). Red before the persistent path exists.

#### D2b — landed (2026-06-30, uncommitted on `b0-3-backend-contract`)

**Mask hoist (byte-exact refactor).** The `[1,1,seq,max_seq_len]` causal mask moved out of
`apply_layer_with_kv_writes` (was one Const per layer) to ONE Const built in
`forward_with_kv_context_impl` and passed into each layer (like RoPE) — `lazy.rs`
`apply_layer_with_kv_writes` now takes a `mask: &LazyTensor` param; the mask data comes from a new
free fn `build_decode_causal_mask(cached_len, seq, max_seq_len)`. Byte-identical across layers by
construction (mask depends only on `cached_len`/`seq`/`max_seq_len`); cuts the per-token re-bind
count on the persistent path from `n_layers` to 1. The existing D1 decode gate
(`forward_with_kv_context_decode_matches_non_cached_forward`) stays green (the born-red for this
refactor).

**`DecodeSession`** (`inference_context.rs`, beside `InferenceContext`). Holds: the optimized
`graph: Arc<RwLock<Graph>>`, the cached `OptimizedGraph`, the D2H-spliced `effective_target` +
`logits_node`, the STABLE data-Const NodeIds (`token_ids_node`, `rope_cos_node`, `rope_sin_node`,
`mask_node`, `kv_nodes: Vec<(k,v)>`), `cached_len_sym`, the **full realized `base_cache`**
(StorageCache; see below), and validity keys (`seq`, `max_seq_len`, `n_layers`, `cache_dtype`).
`is_valid_for(...)` is a cheap field compare; `realize_token(device, DecodeTokenData, sym_env)`
does the per-token prebuilt realize. Held by the generate-loop owner as `&mut Option<DecodeSession>`
(NOT on the immutable `LlamaModel`).

**Deviation from §2/§3 (necessary — a real gap the design's `ctx.insert`-only re-bind missed).**
The prebuilt-realize seam SKIPS `build_const_cache` — so it does **not** upload the model **weight**
Consts (`token_embedding`, per-layer projections, output) that live in `graph.storage_map`, not in
`ctx.persistent`. The naive "re-bind data Consts into `ctx` each token" (design §2 option b) would
fail on the first weight Const (`Const node NodeId(..) not in input cache`). Fix: capture the FULL
`StorageCache` the first realize builds (weights + KV + initial data) and HOLD it on the session
(`base_cache`); each subsequent token clones it and overwrites ONLY the 4 data-Const entries
(token-ids / RoPE cos+sin / mask). Added `pipelined_bridge::prebuild_optimized_env_capturing_cache`
+ `InferenceContext::prebuild_optimized_capturing_as_with_env` (the capturing prebuild returns the
cache alongside `effective_target` + `OptimizedGraph`); `dispatch_with_plan_retry_capturing` now
returns the final cache. This is strictly stronger than §2's plan: the const-cache walk is skipped
**entirely** on every reuse token (a secondary win the design flagged), and the KV Arcs live in
`base_cache` (mutated in place by `Op::WriteSlice` — the same accumulation as D1).

**Data-Const re-bind mechanism (§2 option b, backend-agnostic).** Per token, host bytes are
recomputed — token-ids = the new token (U32 `[seq]`), RoPE cos+sin via `build_rope_tables(base,
cached_len, seq, head_dim)` (`[seq,head_dim]` F32), mask via `build_decode_causal_mask(cached_len,
seq, max_seq_len)` (`[1,1,seq,max_seq_len]` F32; the `-inf` boundary `k > cached_len + q` shifts
per token) — and uploaded to a device-resident `fuel_memory::Storage` Arc via a new
`pipelined_bridge::upload_host_buffer_to_device(device, HostBuffer)` (CPU: wraps bytes; non-CPU:
the transient `Op::Const → Op::Copy{target}` H2D, mirroring `build_const_cache`'s non-CPU arm). The
NodeIds stay stable (the held graph's Const nodes); only the `base_cache` entries change. **Q1
resolved as (b), generalized: base_cache-overwrite instead of `ctx.insert`.** CPU is the born-red
bed; the non-CPU upload arm is wired but GPU-unverified (a later verify).

**`forward_with_kv_context_persistent(tokens, cache, ctx, session)` control flow.** (1) `seq != 1`
→ drop the session (removing any ctx bindings) + fall back to `forward_with_kv_context` (D1
rebuild). (2) `seq == 1` with a session whose validity keys mismatch → drop it (fall through to
build). (3) `seq == 1`, no session → **build**: construct the graph with placeholder data + KV
Consts (`const_placeholder_like` + `ctx.insert`), `prebuild_optimized_capturing_as_with_env`
(optimize + realize ONCE, `OPTIMIZE_CALLS` +1), capture `base_cache`, remove the transient ctx
bindings (they live in `base_cache` now), populate `session`, bump `cache.cached_len` + versions.
(4) `seq == 1`, valid session → **reuse**: recompute the 4 data Arcs, `session.realize_token(...)`
(SKIP optimize, `OPTIMIZE_CALLS` unmoved), bump state. A `TopologyChanged` on reuse drops the
session + falls back to D1 this token (rebuilds next token). Held Consts persist across tokens (in
`base_cache`); removed on session drop.

**Born-red tests** (`lazy.rs` `generate_tests`, CPU):
- `forward_with_kv_context_persistent_plan_once_matches_d1` — prefill (seq 3) then 4 decode tokens
  through the persistent path, with the D1 reference logits captured in a SEPARATE pass first (so
  D1's per-token re-plans don't pollute the measured window). Asserts (a)
  `optimize_calls_thread_local()` delta == 1 across all decode tokens; (b) each token's logits
  **exactly** `==` the D1 cached path; (c) the held graph `len()` stable from token 2 onward. Also
  asserts prefill does NOT build the session.
- `forward_with_kv_context_persistent_invalidates_on_non_decode_step` — a mid-stream `seq!=1` step
  drops the session (fallback), and the next `seq==1` token rebuilds a NEW graph Arc + byte-matches
  the D1 path over the identical history.
- **Red→green observed:** first red = compile failure (`DecodeSession` / method / `graph_node_count`
  absent); progressed through logical reds (session not built on token 1 — a control-flow bug where
  `seq==1 && no-session` wrongly hit the fallback; then `Const node not in input cache` — the weight
  gap above, fixed by `base_cache`; then optimize-count `5 != 1` — a TEST-harness pollution from the
  lockstep D1 reference, fixed by capturing D1 in a separate pass); then green. The 5→1 red proved
  (a) is not vacuous: the fallback path DOES optimize per token, only the reuse path skips.

**Verified:** `cargo test -p fuel-core --lib` = **1286 passed / 0 failed / 10 ignored** (was 1284 at
HEAD; +2 D2b tests). Regression-critical gates confirmed green:
`forward_with_kv_context_decode_matches_non_cached_forward`, all 8 `forward_with_kv_context*`
tests, the D2a `d2a_prebuilt_realize_skips_optimize_and_does_not_grow_graph`, generate/llama2c/
spec-decode. Perf (the ~1.8×/token) is a later GPU verify — NOT gated here (CPU planning is a
smaller fraction of CPU compute; correctness + the optimize-skip COUNT are the CI gates per §10).

### D2c — generate-loop integration + multi-token byte-exact
- Wire the held `session` into the (test/caller) generation loop; ensure invalidation on
  prefill/seq-change/cache-clear (§5).
- **Born-red test (fuel-core, CPU):** a full generate of N tokens (prefill seq>1 then N seq==1
  decode steps) via the persistent path produces the **byte-identical token sequence + logits**
  vs. the D1 per-token path, AND `OPTIMIZE_CALLS` == (1 prefill + 1 decode build) regardless of N.
- The existing `forward_with_kv_context_decode_matches_non_cached_forward` must STILL pass
  (regression guard) — it exercises the D1 path, which D2 leaves intact.

#### D2c — landed (2026-06-30, uncommitted on `b0-3-backend-contract`)

**Wiring (`fuel-core/src/lazy.rs`, `impl LlamaModel`).** The plain-decode generate loop
`generate_streaming_with_kv_context` (`~lazy.rs:6251`) now holds ONE loop-internal
`let mut session: Option<DecodeSession> = None;` across the whole generation and routes BOTH the
prefill AND every per-token decode step through
`forward_with_kv_context_persistent(&[..], &mut cache, &mut ctx, &mut session)` (was bare
`forward_with_kv_context`). `generate_with_kv_context` (`~lazy.rs:6303`) is a thin wrapper over the
streaming loop, so it is now on the persistent path too — no separate edit. **Public signatures
unchanged** (the session is loop-internal state; callers see the same API).

**Prefill handling — routed through persistent, behaviour identical.** The prefill call
(`seq>1`) goes through `forward_with_kv_context_persistent`, which for any non-`seq==1` step
immediately drops the session (a no-op when it's `None`) and delegates to `forward_with_kv_context`
— i.e. the D1 rebuild path, byte-for-byte the pre-D2c prefill, and it does NOT build the session.
So the first *decode* token (the first `seq==1` call) is what builds the held graph (optimize once)
and every subsequent decode token reuses it (skips optimize). Chose "route prefill through
persistent" over "keep prefill on the bare entry" because it is the cleaner single call site (one
`forward_with_kv_context_persistent` for the whole loop) and the internal fallback makes it provably
identical to the bare prefill.

**Left on the rebuild path (unchanged, by design §7):**
- `generate_streaming_spec_with_kv_context` (`~lazy.rs:6372`) — spec-decode's verification step uses
  `forward_with_kv_context_all_positions` (multi-token, `seq>1`); the held graph is shape-keyed to
  `seq==1`, so this stays on `forward_with_kv_context`. (Its per-token draft/target steps also stay
  on the bare entry — spec decode threads two models + truncation, out of D2c scope.)
- `PhiModel::generate_streaming_with_kv_context` / `generate_with_kv_context` (`~lazy.rs:7206/7257`)
  — the second model is D4; left on the rebuild path.

**Born-red test** (`lazy.rs` `generate_tests`, CPU):
`generate_loop_persistent_byte_exact_and_plans_once`. Drives an explicit persistent generate loop
(mirroring the wired production loop: hold `session`, `forward_with_kv_context_persistent` for
prefill + every decode step) over a 3-token prompt + **N=5 greedy decode tokens**, against a
SEPARATE D1 reference loop (bare `forward_with_kv_context` + the identical greedy `sample_logits`)
captured FIRST in its own pass (so D1's per-token re-plans don't pollute the measured optimize
window). Asserts: **(a)** the generated token sequence is **byte-identical** over N tokens — greedy
argmax diverges on ANY per-token logit drift, so an exact N-token match is a strong end-to-end
guard; **(b)** each step's logits are **exactly `==`** the D1 cached path (bit-exact, NOT epsilon);
**(c)** `optimize_calls_thread_local()` bumps **exactly 2** across prefill + N decode (1 prefill
fallback + 1 decode-session build) regardless of N — the reuse tokens skip optimize. It also drives
the REAL `generate_with_kv_context` wrapper and asserts its token sequence matches the reference
(confirms the wiring, not just the entry). Asserts the session is `None` after prefill and `Some`
after the loop. **Red→green observed:** a temporary probe routing the decode step through
`forward_with_kv_context` (D1) failed (c) `6 != 2` (1 prefill + 5 decode re-optimizes) — proving (c)
is not vacuous and the optimize-skip is exactly what the wiring buys; restoring the persistent
decode call → green.

**Regression guards confirmed green** (in the full suite): `generate_greedy_appends_tokens`,
`generate_with_kv_context_matches_legacy_generate`,
`generate_streaming_with_kv_context_fires_callback_per_token` / `_stops_on_eos`,
`forward_with_kv_context_decode_matches_non_cached_forward` (the D1 gate),
`forward_with_kv_context_persistent_plan_once_matches_d1` (D2b),
`forward_with_kv_context_persistent_invalidates_on_non_decode_step` (D2b), the D2a prebuilt-seam
test, and the spec-decode tests (untouched path).

**Perf scaffold (optional, delivered `#[ignore]`'d, NOT gated).**
`generate_loop_persistent_bench_scaffold` shows the A/B shape (D1 rebuild loop vs. D2 persistent
loop over N=64 seq==1 tokens on the tiny CPU model) and prints per-token wall-clock + ratio, with NO
timing assertion. The real ~1.8×/token measurement is a **manual live-GPU run on a realistic
model** (CPU understates the win — planning is a smaller fraction of CPU compute); the scaffold's
A/B shape is the template to port to a live-GPU harness (N≥64, realistic model, one live suite at a
time per CLAUDE.md).

**Verified:** `cargo test -p fuel-core --lib` = **1287 passed / 0 failed / 11 ignored** (was 1286
passed / 10 ignored at the D2b HEAD `ca518525`; +1 D2c born-red test, +1 D2c ignored bench
scaffold). CPU-only; the wall-clock win is a later GPU verify (the ignored CPU scaffold observed
≈1.87×/tok on the tiny model — indicative only). **Not committed** (per prompt).

### D2d (optional follow-up) — skip `insert_safety_copies` on the prebuilt path
- Only if the D2b/c profile shows `insert_safety_copies` is a measurable per-token cost on the held
  graph. Add an `OrderSource`/`realize_inner` opt-out asserting the graph is already safety-copied.
- Born-red: a counter on `insert_safety_copies` invocations, asserting 0 on prebuilt re-realizes.

---

## 10. Verification plan

- **Byte-exact — against the D1 *cached* path, NOT against non-cached.** Important framing nuance
  verified from the gate: `forward_with_kv_context_decode_matches_non_cached_forward`
  (`lazy.rs:7856`) compares the D1 cached decode to a non-cached forward with an **epsilon
  tolerance** (`diff < 5e-3 || rel < 1e-2`, `lazy.rs:7921`), *because the gemm accumulation order
  differs* between the two (cached prefix + 1 fresh row vs. one length-`total_seq` tensor) — i.e.
  D1-vs-non-cached is NOT byte-exact and never claimed to be. D2's claim is stronger and exact:
  D2 reuses the **same plan → same kernel sequence → identical bytes** as the D1 *cached* path on
  the same inputs. So every D2 born-red test asserts **`Vec<f32>` `==` (exact equality) vs. the D1
  cached path** (`forward_with_kv_context`), and SEPARATELY keeps the existing epsilon gate
  (D1-vs-non-cached) as a regression guard. Do NOT assert D2 byte-exact vs. non-cached — that would
  fail on the same gemm drift the D1 gate tolerates. CPU-first (the born-red bed); then GPU-verify
  on CUDA + Vulkan after behaviour lands (one live suite at a time, per CLAUDE.md /
  environment-discipline).
- **Optimize-skip:** the `OPTIMIZE_CALLS` counter asserts exactly-once-per-session, not per token.
- **1.8× (perf):** a decode micro-benchmark (N≥64 seq==1 tokens on a small model) comparing
  wall-time/token of `forward_with_kv_context` (D1, re-plan) vs.
  `forward_with_kv_context_persistent` (D2, plan-once). Expect ≈1.8× on a backend where planning is
  ~45% of token time (CUDA decode per §1 of the spec). Run on the live GPU after correctness lands;
  CPU will show a smaller ratio (CPU planning is a smaller fraction of CPU compute). **Perf is a
  verify-after gate, not a born-red test** (timing tests are flaky in CI; assert the optimize-skip
  COUNT in CI and measure the ratio manually).
- **No node growth:** assert graph `len()` stability across re-realizes (catches an accidental
  re-splice / re-insert).

### Cadence (CLAUDE.md)
Born-red TDD first per PR; `-p fuel-core` / `-p fuel-dispatch` only (never workspace-wide); one
cargo at a time; never-panic (`TopologyChanged` + unbound-sym surface typed errors, not panics);
lazy-only. GPU-verify (CUDA + Vulkan) after each behaviour-touching PR.

---

## 11. Key risks + OPEN QUESTIONS (resolve before implementing)

- **Q1 — device-resident data Consts** *(verified-feasible; decision = which option)*. §2: re-bind
  token/RoPE/mask as `const_placeholder_like` + `ctx.insert` of a **device-resident** Arc
  (option b, recommended) vs. in-place-mutate graph `Op::Const` bytes + per-token H2D (option a).
  (b) keeps the const-cache walk skippable. The per-token device-overwrite helper EXISTS
  (`CudaStorageBytes::write_from_host`, `byte_storage.rs:294`; `VulkanBackend::write_bytes`), so (b)
  is implementable; the open part is the small `DecodeSession` glue that picks the right write path
  per backend. On CPU there's no issue (slice copy). **CireSnave: confirm (b).**
- **Q2 — entry-point shape.** §6: sibling `forward_with_kv_context_persistent(... session)` (B,
  recommended, easy A/B test) vs. a `decode_step` method (A).
- **Q3 — selector/lookup caching** *(verified safe to cache)*. §1: `production_selector_for`
  (`pipelined_bridge.rs:1131`) builds a fresh `ChainedSelector` Arc from `Device` + the cached Judge
  oracle each call — **no per-realize state**, so caching it on `DecodeSession` is safe. For the
  branchless decode graph `streaming_pick_for` returns `None` and the selector is never consulted
  anyway, so caching vs. rebuilding is a micro-optimization; **prefer rebuild-per-token for D2
  simplicity** (one Arc alloc), cache later if a profile shows it.
- **Q4 — does `insert_safety_copies` fire on the decode graph?** §1/§4/D2d: if the
  WriteSlice→attention edge ordering leaves no conflicting readers, it never inserts and the
  per-token write-lock cost is just an ordering derivation. Measure; decide whether D2d is worth it.
- **Q5 — `TopologyChanged` on the prebuilt path.** §4/§5: recommend invalidate-and-rebuild the whole
  `DecodeSession` (simplest, rare event) vs. re-optimize-in-place the already-prepared graph.
- **Q6 — session reuse across `KvCache::clear()`.** §5: if the same `KvCache` object is reused for a
  new prompt (Arcs persist, `cached_len` resets), the session COULD be reused; D2 conservatively
  rebuilds on clear. Confirm whether any caller reuses a cache across generations.
- **Q7 — `cached_len` as an affine of `seq`.** D1 binds `SymId(0) = cached_len` and the mask
  encodes `cached_len + q`. For seq==1 this is fine. The held graph fixes `seq==1`; if a future
  variant wants seq>1 persistent, the mask's `cached_len + q` per-row pattern must come from the
  re-bound mask Const (it does) — confirm nothing else bakes `seq`.
- **Risk — graph identity assumption.** The entire optimize-skip rests on D1's "structurally +
  shape-identical every token". The born-red node-`len()`-stability assertion is the guard; if any
  builder (token-ids, RoPE, mask) sneaks a `cached_len`-dependent shape in, the cached
  `OptimizedGraph` (stamps) goes stale silently. The node-count + byte-exact tests catch it.
- **Risk — stamp staleness across topology.** Reusing `optimized.generation` is correct only while
  the live `SystemTopology` generation matches; the executor's chunk-boundary check enforces this
  and surfaces `TopologyChanged` → Q5 handles it.
```

---

## D4 — landed (2026-06-30, uncommitted on `b0-3-backend-contract`)

Ported the D1 (input-independent decode graph) + D2b (held session + per-token data re-bind) + D2c
(generate-loop wiring) transform onto the **second model, `PhiModel`** — which the D1 author left on
the sliced rebuild path as a mechanical follow-up. Reuses ALL the model-agnostic machinery unchanged
(`DecodeSession`, `DecodeTokenData`, `DecodeSession::realize_token`,
`InferenceContext::prebuild_optimized_capturing_as_with_env`,
`pipelined_bridge::{realize_one_prebuilt_env, upload_host_buffer_to_device, optimize_calls_thread_local}`,
`LazyTensor::write_slice_dyn`, and the module-level `build_decode_causal_mask`). CPU-verified
byte-exact; the wall-clock win is the same later live-GPU verify D2c documents.

**How Phi differs from Llama (and how the transform adapted).** Phi's decode block is NOT a copy of
Llama's, so the transform tracked Phi's actual structure:
- **Parallel attention + MLP over a single shared pre-block `LayerNorm`** (not RMSNorm), with the
  parallel residual `x + attn_out + mlp_out`. The D1 transform is orthogonal to this — it only
  rewrote the KV-write + attention-extent + mask, so the parallel MLP branch and the shared
  `x_norm` were left exactly as-is.
- **Partial RoPE** (`rotary_dim < head_dim`; only the first `rotary_dim` head entries rotate, via
  `partial_rope`). The RoPE tables are sized `[seq, rotary_dim]`, NOT `[seq, head_dim]` — so both
  the placeholder Const shape (`build_and_realize_first_decode_token`) and the per-token host-byte
  recompute (`build_token_rope_mask_arcs`) use `cfg.rotary_dim` in `build_rope_tables(...)`. This
  was the single most error-prone adaptation vs. the Llama helper (which uses `head_dim`).
- **Bias on every projection** (Q/K/V/dense/fc1/fc2) + an **optional output bias**. The output-bias
  branch (`weights.output_bias` → `broadcast_add`) is reproduced verbatim in the persistent build so
  the held logits root is byte-identical to the D1 path's.
- **No GQA** (`n_kv_heads == n_heads`): the KV cache carries `n_heads`; the KV cache shape and the
  `write_ranges` axis-1 extent use `cfg.n_heads` (Llama uses `cfg.n_kv_heads`).
- **QKV packing** (`PhiQkv::Split` vs `Packed`): untouched — it lives inside
  `apply_layer_with_kv_writes` before the KV-write, and the transform did not move it.

**Phi-D1 transform (`PhiModel::apply_layer_with_kv_writes`).** Made byte-identical across tokens:
KV write offset `write_slice` (concrete `cached_len`) → `write_slice_dyn` at
`DynScalar::Sym(cached_len_sym)` on axis 2 (width `seq`); dropped `slice(2, 0, total_seq)` — attend
the FULL fixed-capacity buffers `[batch, n_heads, max_seq_len, head_dim]`; the per-layer mask Const
is hoisted to ONE shared `[1,1,seq,max_seq_len]` causal mask built in the forward
(`build_decode_causal_mask`, `-inf` where `k > cached_len + q`, masking future positions AND the
zero-init stale tail). `forward_with_kv_context` binds `cached_len_sym = SymId(0)` + a per-pass
`SymEnv` and realizes via `realize_one_as_with_env`. Numerically identical to the sliced form
(masked positions contribute 0).

**Phi persistent path.** Added `PhiModel::forward_with_kv_context_persistent(tokens, cache, ctx,
&mut Option<DecodeSession>)` + the three private helpers (`build_and_realize_first_decode_token`,
`rebind_and_realize_prebuilt`, `build_token_rope_mask_arcs`, `drop_decode_session`) mirroring the
LlamaModel siblings: seq!=1 / stale-keys / `TopologyChanged` → drop session + D1 rebuild fallback;
first seq==1 token → build the held graph with stable re-bindable data Consts + capturing prebuild
(OPTIMIZE +1); subsequent seq==1 tokens → recompute host bytes (token-ids / partial-RoPE tables at
`position = cached_len` / shifted mask) + realize via the prebuilt seam (SKIP optimize).

**Generate-loop wiring.** `PhiModel::generate_streaming_with_kv_context` now holds ONE loop-internal
`Option<DecodeSession>` and routes prefill + every decode step through
`forward_with_kv_context_persistent` (prefill seq>1 falls back WITHOUT building the session; first
decode token builds; the rest reuse). `generate_with_kv_context` delegates to the streaming loop, so
it is on the persistent path too; public signatures unchanged. (No plain-decode spec loop exists for
PhiModel; there was nothing left on a rebuild path.)

**Born-red tests** (`lazy.rs` `phi_kv_context_tests`, CPU):
- `phi_decode_matches_non_cached_forward` — Phi-D1 correctness: prefill(3) + decode(1) through the
  input-independent graph vs. a monolithic prefill over the same 4-token history (PhiModel has no
  non-cached `forward`, so the monolithic prefill is the reference, matching the existing
  `phi_kv_context_decode_consistent_with_monolithic_prefill`). Within the existing O(ε) gemm band.
- `phi_persistent_plan_once_matches_d1` — Phi-D2: prefill + 4 DISTINCT decode tokens; asserts
  **(a)** `optimize_calls_thread_local()` bumps EXACTLY 1 across all decode tokens, **(b)** each
  token EXACTLY `==` the Phi rebuild path (bit-exact), **(c)** held graph node `len()` stable from
  token 2.
- `phi_generate_loop_persistent_byte_exact_and_plans_once` — end-to-end: an explicit persistent
  generate loop (N=5 greedy) vs. a separate D1 reference loop; asserts byte-identical token
  sequence, per-step logits `==`, `optimize` bumps EXACTLY 2 across prefill + N decode; also drives
  the real `generate_with_kv_context` wrapper.

**Red→green observed.** A temporary probe forcing `forward_with_kv_context_persistent` onto the D1
rebuild path made `phi_persistent_plan_once_matches_d1` fail (optimize count per-token, not once) and
`phi_generate_loop_persistent_byte_exact_and_plans_once` fail `optimize EXACTLY twice ... 6 != 12`
(N=5 → +6 re-optimizes instead of +2) — proving assertion (c) is non-vacuous. Restoring the
persistent path → green.

**Verified:** `cargo test -p fuel-core --lib` green with the 3 new Phi tests; the Llama D1/D2a/D2b/D2c
gates + all pre-existing Phi tests (`phi_kv_context_decode_consistent_with_monolithic_prefill`,
`phi_generate_with_kv_context_greedy_is_deterministic`,
`phi_forward_with_kv_context_rejects_invalid_cache`) stay green. CPU-only. **Not committed** (per
prompt).
