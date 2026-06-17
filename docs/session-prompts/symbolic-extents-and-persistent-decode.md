# Symbolic extents + persistent decode — design

**Status**: design, agreed 2026-06-16 (converged through architect discussion). This is the
concrete mechanism for runtime-dependent dimension extents and the path to the persistent
decode graph. It **refines the Phase D section** of
[`plan-is-graph-rebuild.md`](plan-is-graph-rebuild.md) (`[8]` load-time build, `[9]` sessions,
`[10]` persistent decode) and is the concrete realization of
[`data-dependent-shapes-design.md`](data-dependent-shapes-design.md) for the *input-determined*
case. **Promote the chosen mechanism into [`03-ir`](../architecture/03-ir.md) once built** (per
03-ir's note that "input-dependent extents are supported … a bounded symbol").

D1 (the `StorageClass` substrate) already landed; this supersedes the old D2/D3/D5 framing.

---

## 1. The problem

Autoregressive decode re-plans every token. The decode-step graph is rebuilt fresh per token
(`LlamaModel::forward_with_kv_context_impl`, `fuel-core/src/lazy.rs`), and `optimize_graph` runs
per realize. The cost (~45% of token time → the lost ~1.8×) is the re-plan, not the compute.

Why it can't trivially be built once: the **attention width grows by one each token**.
`total_seq = cached_len + seq` drives the K/V slice extent, the scores shape `[seq, total_seq]`,
and a host-built `[seq, total_seq]` causal mask (lazy.rs:5039–5100). So even though the *ops* are
identical every token, one *dimension's size* differs every token → a differently-shaped graph →
a different plan.

The fix is to make that one dimension a **runtime value over a fixed-capacity buffer**, so the
graph's structure *and shape-expressions* are identical every token and the plan is reused.

---

## 2. Design philosophy: base map correct, fusion is an optimization

This maps onto fuel's existing **base-map ↔ optimized-graph** split (03-ir):

- **Base map** (canonical, primitive, always correct, portable, runnable): attention stays
  `matmul → mask → softmax → matmul`. With symbolic extents (below), the KV-length axis is a
  bounded symbol; the decomposed form is correct and runnable as-is on every backend — `matmul`
  already takes its dims at runtime, `softmax` reduces over the concrete extent, buffers allocate
  to the bound. **Nothing must become `FlashAttn` for it to work.**
- **Optimizer**: a pathfinder recognizes the `matmul → mask → softmax → matmul` pattern and *adds*
  an `Op::FlashAttn` **alternative arm** (Phase A's `Op::Branch` multi-path), consuming the
  symbol as its `k_len`. The ranker/route-picker prefers the flash arm where a flash kernel exists
  (CUDA) and falls back to the decomposed arm elsewhere (CPU/Vulkan until they have a fused
  kernel). **Fusion is a speed lowering, never a correctness prerequisite.**

Consequences: cross-backend is never a blocker (decomposed base map runs everywhere; flash is
additive per backend); the flash arm is validated *against* the base map (the Judge), so we never
trust fusion for correctness, only speed. The **persistent-decode / 1.8× win comes from the
symbolic foundation** (input-independent graph → plan once), *not* from the fusion; fusion adds the
per-token compute/memory win (no materialized scores/mask) on top.

---

## 3. The foundation

### 3.1 The primitive: `SymId` + `SymEnv`

A standard symbolic-shape environment (cf. MLIR/TVM shape vars resolved by a binding map).

- **`SymId`** — an interned identifier (newtype over `u32`/`u64`). The stable, **serializable**,
  session-independent identity of a runtime value. (We keep ids, not pointers: pointers can't
  serialize for the base map, and a graph-embedded cell would clobber across concurrent sessions —
  see the discussion log.)
- **`SymEnv`** — the registry: `SymId → usize`, with `bind(sym, value)` / `get(sym)`. It is a
  **per-forward-pass input**, a sibling of the `StorageCache` that already carries tensor data
  inputs. The graph carries `SymId`s (shared, immutable, serializable); the `SymEnv` is supplied
  per realize. This is `[8]` exactly: the graph is input-independent; symbol bindings are *part of
  the per-pass input*, flowing through alongside tensor data.

There is **no language-level aliasing**. Two places "share a value" by holding the **same
`SymId`** and each reading `env.get(sym)` on demand — single source of truth + read-on-demand, not
cache-and-invalidate. Nothing is copied, so nothing needs propagation/invalidation.

### 3.2 Two carriers that reference a `SymId`

- **`Extent` — for dimensions:**
  ```
  enum Extent { Scalar(usize), Range { min: usize, max: usize, sym: SymId } }
  ```
  `Scalar` is a build-time constant (no sym — two constants that must match already match by being
  the same number; a *runtime* dimension always needs a capacity, so it is a `Range`). `Range`
  carries the capacity bounds **and** the `sym` for runtime resolution + unification.
- **`DynScalar` — for op params** (offsets, positions, lengths that aren't dimensions):
  ```
  enum DynScalar { Concrete(usize), Sym(SymId) }
  ```
  No bounds; just concrete-or-symbol. Decode needs this *now* for the `WriteSlice` offset, the
  RoPE position, and `flash`'s `k_len` — all functions of `cached_len`. (The persistent graph
  can't bake the `WriteSlice` offset, so the param carrier is part of step 1, not just `Extent`.)

Both resolve through the one `SymEnv`.

### 3.3 Annotated `Shape` — `dims()` does **not** change

`Shape` is used ~2,000× across 250 files via `dims() -> &[usize]`; that contract stays. We change
`Shape` **additively**, not by swapping its element type:

```
struct Shape {
    dims: SmallVec<[usize; 6]>,          // unchanged — the BOUNDS; dims() -> &[usize] borrows this
    dynamic: Option<SmallVec<[DynAxis]>>, // sparse: which axes are Range + their min/sym
}
```
(`DynAxis` records `{ axis, min, sym }`; `max` is the corresponding `dims[axis]` bound.)

- **`dims() -> &[usize]`** — unchanged, zero copy. Returns the **bounds**: a scalar's value, or a
  range's `max`/capacity. This is the correct value for the ~2,000 sizing/striding/iteration sites
  (allocate capacity, walk the buffer). It is **not** a lie — it's the capacity bound; the live
  value is a *different fact* obtained elsewhere.
- **`extent(i) -> Extent`** — the enum *view*, computed from `(dims[i], dynamic)`. For
  shape-inference and symbol-aware code.
- **`resolve(&env) -> Shape`** (or `concrete_dims(&env) -> DimVec`) — owned, reads `env.get(sym)`
  at call time, substitutes concrete values. The opt-in realize-time "give me the concrete shape"
  path. Always current (never cached).
- `from_dims(&[usize])` is unchanged — the all-`Scalar` constructor (`dynamic: None`). Add a
  `from_extents`/builder for symbolic construction.
- `Hash`/`Eq`/`Debug` include `dynamic`, so a symbolic shape is a distinct shape that plans once;
  concrete shapes (`dynamic: None`) hash exactly as today.

The internal rep change (tuple → named struct) is contained to `shape.rs` (the field is private;
external code constructs via `from_dims`/`From`).

### 3.4 `Layout` inherits `Extent` for free

`Layout { shape: Shape, stride: StrideVec, start_offset: usize }` **embeds a `Shape`**, so when
`Shape` gains `Extent`, `Layout` inherits it with no duplication. They are *layered*, describing
orthogonal facts:

- **`Shape`/`Extent`** = logical sizes — *how many* elements per axis (concrete or symbolic-live).
- **`Layout`** = `Shape` + strides + offset = *how to step through* the buffer.

For a symbolic axis the **stride stays concrete** (the physical step in the fixed-capacity buffer,
a build-time constant), while the **extent is symbolic** (the live count). Those are exactly the
two halves a kernel needs to walk the *live prefix of a capacity buffer* — stride = how far per
element, extent = how many live elements — so they complement rather than collide. The existing
`Shape`↔`Layout` coherence rule extends to "agree on the `sym` too"; view ops
(`narrow`/`slice`/`transpose`) derive the `Layout` from the input `Shape`, so the symbol
propagates consistently (no new divergence risk beyond the one already managed for concrete dims).

### 3.5 Masks are ops, not `Extent`

`Extent` exposes the **scalar** (resolve to the live `usize`, plus `min`/`max`/`is_dynamic`). It
does **not** materialize masks — a mask is a device-side validity *tensor*, an **op's** job. The
"some kernels want a length, some want a mask" split is a **lowering** decision:

- length-consuming kernel (`flash`'s `k_len`) → feed `extent.resolve(env)` directly;
- mask-consuming kernel → a `causal_mask(seq, extent, offset)` op consumes the scalar extent and
  produces the mask tensor (on the right device, cacheable, optimizable).

Three orthogonal concepts, none duplicating: **`Extent`** (logical size, a number), **mask** (a
validity tensor an op builds from that number), **`Layout`** (physical walk).

---

## 4. Resolution & lifetime semantics

- **`dims()` returns immutable bounds.** Bounds are build-time constants (the KV axis is
  `max_seq_len`, fixed at build). The reference `dims()` hands out can never go stale; the thing
  that changes per pass is not behind it.
- **The live value lives in one `SymEnv` entry, read on demand.** No copies → no propagation, no
  invalidation. `env.bind(sym, v)` overwrites one entry; consumers re-read the single source.
- **Write-once per forward pass.** A `sym` is written exactly once per (pass, session):
  - *Input-determined* (KV length, prompt length, batch) — bound up-front by the caller, immutable
    for the whole pass. **This is all of decode.**
  - *Data-determined* (NonZeroIndices count, MoE counts, data-dependent top-k) — filled *during*
    the pass by its producing op, then read by consumers. Still write-once-read-many.
- **Presence ⇒ produced.** Because of write-once, "`sym` present in the env" is provably "its
  producer has run." A good *runtime* readiness signal / assertion (and, in an async runtime, a
  potential await). But it **detects**, it does not **prevent**, a bad order — the **build-time
  dependency edge** is what schedules a data-determined consumer after its producer (without it, a
  single dispatch thread can deadlock). Decode never hits this: `cached_len` is bound before the
  first op runs, so it is always present when read.
- **Scope axis** (recorded so the API anticipates it): syms have a lifetime — **pass-scoped**
  (`cached_len`, re-bound each pass), **session-scoped** (batch size, bound once per session,
  constant across its passes), **mid-pass** (data-determined). For step 1 keep it simple: the env
  is per-pass and the session re-supplies every sym each pass. Later, the `SymEnv` may cache
  session-scoped syms to avoid re-binding — the API should not preclude it.

---

## 5. The decode application

Per token (one forward pass), `cached_len` is the single input-determined symbol (call it `s`):

- **KV-length dim** of the (capacity) K/V buffers → `Extent::Range { min, max: max_seq_len, sym }`.
  K-length and V-length **unify to the same `sym`** (§6).
- **`WriteSlice` offset** (where the new token's K/V append) → `DynScalar::Sym(sym)`.
- **RoPE position** → `DynScalar::Sym(sym)` (or `s` directly).
- **`flash` `k_len`** = `cached_len + seq`. (Affine over `s`; for now the binder computes it and
  binds a derived sym, or the op composes `s + seq` from the concrete `s`.)
- **Attention arms:** base-map arm = `matmul → causal_mask(s) → softmax → matmul` (correct
  everywhere); fused arm = `Op::FlashAttn` with `k_len` (CUDA), added by the pathfinder, chosen by
  the picker. Both consume `s`; the mask arm additionally has the `causal_mask` op.

The graph is then structurally identical every pass; only the data inputs (token, KV contents,
RoPE tables) and the `SymEnv` binding of `s` change. `optimize_graph` runs once; realize re-binds
+ dispatches.

---

## 6. Symbol unification

Shape inference allocates a fresh `SymId` where a dynamic extent originates, and **unifies** syms
that must be equal: K-length ≡ V-length (same cache region) → the same `sym`; an op whose output
length equals an input length propagates the input's `sym`. Unification is by id equality. Two
distinct `Range`s with different syms are *not* interchangeable even at equal bounds. This is what
the side-table option could not do without `NodeId` plumbing and the pointer option could not
serialize — the id-in-`Shape` form does it locally and on disk.

---

## 7. Vocabulary (fixed, to avoid collision)

- **forward pass** / **realize** / **decode step** — one time through the model. The symbol
  lifetime unit for pass-scoped syms.
- **run** (reserved, Phase C) — the op-sequence between two decision points; the dispatch unit.
  *Many runs per forward pass.* A sym is constant across all runs within a pass.

---

## 8. Build order

D1 (`StorageClass`) — **done.** Then:

1. **Symbolic-extent foundation** (this step). `SymId` + `SymEnv`; `Extent`; `DynScalar`;
   annotated `Shape` (`dims()` unchanged + `extent()`/`resolve(&env)`, `Hash`/`Eq`/`Debug` updated,
   symbolic constructor); `Layout` inherits via its embedded `Shape`. Foundational; no consumers
   yet. Born-red test: a `Shape` with a `Range` axis returns the bound from `dims()`, the `Range`
   from `extent()`, the concrete value from `resolve(env)`; two dims sharing a `sym` resolve
   together; `Hash`/`Eq` separate symbolic from concrete; `from_dims` unchanged.
2. **Symbolic causal-mask op** — `causal_mask(seq, extent, offset)` so the decomposed attention
   base map is self-contained over `s` (no host-rebuilt per-token mask). Keeps the base map
   correct + persistent.
3. **Attention-fusion pathfinder + decode via flash** — recognize `matmul → mask → softmax →
   matmul`, add the `Op::FlashAttn` arm with `k_len = s`; ranker/picker prefer it on CUDA, fall
   back to the decomposed arm. Verify numerics against the base map (Judge); confirm Vulkan/CPU
   run the decomposed arm.
4. **Persistent decode wiring** (`[8]`+`[9]`+`[10]`) — build the decode-step graph once; the
   generate loop re-binds data + the `SymEnv` per pass and re-realizes the *same* graph;
   `optimize_graph` runs once; session-state storage keyed for concurrent sessions. The 1.8×
   reaches production. Largely *falls out* of step 1, since the graph is already input-independent.

---

## 9. Recorded future (design the API general; do not build the consumers yet)

- **Data-determined syms** — `NonZeroIndices`, MoE per-expert counts, data-dependent top-k:
  producer-filled mid-pass + the producer→consumer **dependency edge** for scheduling. (Build with
  the data-dependent-shapes program; the foundation already serves them.)
- **Batch / ragged** — dynamic batch size, per-sequence lengths.
- **Session-scoped syms** — bound once per session (vs per pass); the `SymEnv` may later cache
  them.
- **Affine sym expressions** — `k_len = cached_len + seq` resolved transitively; start with the
  binder computing derived values.
- **Shape-arithmetic discipline** — building a *new* shape from a dynamic axis must go through the
  `extent`/symbolic path, not `dims()`-bounds (else it bakes the `max` and loses the symbol).
  Enforce at the few op-builder sites that do shape math over a dynamic axis; decode never hits it.
- **Promote to 03-ir** once built (the bounded-symbol mechanism it already gestures at).
