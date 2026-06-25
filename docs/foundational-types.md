# The atoms — Fuel's foundational types

**Audience**: a new contributor (human or a fresh model instance) who needs to read
Fuel's code without drowning. The architecture set (`docs/architecture/`) says *what*
Fuel is and *why*; it is deliberately free of type signatures. This doc is its
concrete companion: *what the basic types actually are and where they live*, so the
code maps onto the vision. Read [14-lifecycle](architecture/14-lifecycle.md) for the
end-to-end flow first; read this to know the nouns.

> File:line references are navigation aids, not contracts — code moves. The *shape*
> below is the durable part.

---

## The one shape to hold in your head

A model is a **lazy DAG**. A `Graph` is an **arena of `Node`s**. A `Node` is exactly
four things:

```rust
struct Node { op: Op, inputs: Vec<NodeId>, shape: Shape, dtype: DType }
```

That's the whole atom. Edges are `inputs: Vec<NodeId>` (a node names its operands by
id). Everything else in Fuel is either **vocabulary those four fields are built from**,
or **machinery that reads the graph** (the optimizer, the executor, the backends).

What a `Node` deliberately does **not** carry: no kernel, no backend, no device, no
"plan." Those are decided elsewhere and (where they belong on the graph) live in
`Graph` *side-tables*, not on the node. Hold onto that — it's the load-bearing absence
that the last section explains.

---

## Layer 0 — `fuel-core-types` (the vocabulary every crate imports)

The bottom crate. Zero backend dependencies. If a type describes a *shape*, a *value*,
a *dtype*, or an *identity*, it almost certainly lives here.

| Type | Kind | Where | What it is |
| --- | --- | --- | --- |
| `DType` | enum | `dtype.rs:14` | logical element type — F32, BF16, F16, I32, U8, … |
| `Shape` | struct | `shape.rs:101` | the dims (+ an optional sparse `dynamic` side for symbolic extents); `dims() -> &[usize]` is the hot path |
| `Extent` | enum | `shape.rs:23` | one axis's size: `Scalar(usize)` (build-time constant) or `Range { min, max, sym }` (a runtime size over a fixed capacity) |
| `Layout` | struct | `layout.rs:24` | `Shape` + strides + offset = *how to walk* a buffer (logical sizes vs physical stepping are layered, not conflated) |
| `Scalar` | enum | `scalar.rs:8` | one typed value (`F32(f32)`, `I32(i32)`, …) — the dtype-erased scalar |
| `OpKind` | enum | `dispatch.rs:52` | the **kernel-dispatch key**. *Not* `Op` — see below |
| `BackendId` | enum | `probe.rs:63` | Cpu / Cuda / Vulkan / Metal — which backend a kernel runs on |
| `DeviceLocation` | enum | `device.rs:17` | which physical device (backend + index) |
| `SymId` | newtype `u32` | `symbol.rs:29` | identity of a runtime value (see "runtime-value primitives") |
| `SymEnv` | struct | `symbol.rs:59` | per-pass `SymId -> usize` binding registry |
| `DynScalar` | enum | `symbol.rs:119` | a scalar op-param that is `Concrete(usize)` or `Sym(SymId)` |

## Layer 1 — `fuel-memory` (the only place real bytes live)

| Type | Kind | Where | What it is |
| --- | --- | --- | --- |
| `Storage` | struct | `lib.rs:82` | realized bytes + dtype; the thing a kernel reads/writes |
| `BackendStorage` | enum | `lib.rs:60` | the per-backend buffer: `Cpu(..)` / `Cuda(..)` / `Vulkan(..)` / `Metal(..)` |

`Storage` is *realized data*. The graph is *unrealized description*. Realization
(running the graph) is what turns the second into the first.

## Layer 2 — `fuel-graph` (the IR, built from Layer 0)

| Type | Kind | Where | What it is |
| --- | --- | --- | --- |
| `NodeId` | newtype `usize` | `lib.rs:117` | an index into the arena; the unit of an edge (`inputs: Vec<NodeId>`) |
| `Op` | enum | `lib.rs:210` | **the op basis**: a closed set of primitive variants + one delegate arm, `Op::Fused(FusedOpId, FusedOpParams)`, to the open fused-op registry |
| `Node` | struct | `lib.rs:1285` | `{ op, inputs, shape, dtype }` — the atom |
| `Graph` | struct | `lib.rs:1348` | the arena of `Node`s + sparse side-tables (`target_backend`, `layouts`, `storage_class`, `storage_map`, …) |
| `Tensor` | struct | `lib.rs:2530` | a **build handle** — `{ graph: Arc<RwLock<Graph>>, id: NodeId }`. The cursor you call `a.matmul(&b)` on; it *is not data*. (`fuel-core`'s `LazyTensor` wraps it for the user API) |
| `FusedOpId` | newtype `u16` | `registry.rs:64` | stable id of a registered fused op (static ids dense from 1; runtime/JIT ids from `RUNTIME_FUSED_BASE = 0x8000`) |
| `FusedOpParams` | enum | `registry.rs:159` | per-instance params for a fused-op node (e.g. `FlashAttn { softmax_scale, causal, … }`) |

`Op` is closed on purpose (a small primitive basis every backend must cover); the
single `Op::Fused` arm is the open extension point, so adding a fused op is a registry
entry + a kernel, never a new `Op` variant.

---

## The runtime-value primitives (the part worth understanding well)

Most dimensions and op-params are known when the graph is built. A few are only known
*per forward pass* — the KV-cache write offset (`cached_len`), the RoPE position, a
fused-attention `k_len`, a data-dependent count. Baking those into the graph forces a
**fresh graph per token**, which is exactly the cost autoregressive decode pays. The
runtime-value primitives let the graph stay fixed while those values change per pass.

The mechanism is a standard symbolic-shape environment (cf. MLIR/TVM shape vars):

- **`SymId(u32)`** — the *interned, stable, serializable* identity of a runtime value.
  It is an **id, not a pointer or a cell**: pointers can't serialize into the base map,
  and a graph-embedded mutable cell would clobber across concurrent sessions. **Equal
  ids denote the same value** — which is how unification works (a KV cache's K-length
  and V-length carry the *same* `SymId`, so they resolve together with no aliasing).

- **`SymEnv`** — the per-forward-pass registry, `SymId -> usize`, with `bind`/`get`. It
  is the **sibling of the tensor-data cache**: the graph carries `SymId`s (shared,
  immutable, serializable); the `SymEnv` supplies their concrete values *per realize*.
  Bindings are **write-once per pass** (a symbol's value is fixed for the pass), so
  "present in the env" provably means "its producer has run."

- **`DynScalar`** — a scalar op-param carrier: `Concrete(usize)` (a build-time constant)
  or `Sym(SymId)` (resolved via the `SymEnv` at realize). Used where a scalar param is
  runtime but *not a dimension* — the `WriteSlice` offset, the RoPE position, flash's
  `k_len`.

- **`Extent`** — the same idea for a *dimension*: `Scalar(usize)` (a static size) or
  `Range { min, max, sym }` (a runtime live count over a fixed capacity bound). The
  capacity (`max`) is a build-time constant the buffer allocates to; the `sym` is the
  live count resolved per pass. Stride stays concrete (physical step in the capacity
  buffer); the extent is symbolic (how many live elements) — the two halves a kernel
  needs to walk the live prefix of a capacity buffer.

The whole point: **there is no aliasing.** Two places "share a value" by holding the
*same `SymId`* and each reading `env.get(sym)` on demand — single source of truth,
read-on-demand, nothing copied, nothing to invalidate. The graph is input-independent
(it carries ids); the per-pass input is the `SymEnv` flowing in alongside the tensor
data. That is what turns "re-plan every token" into "plan once, re-bind per token."

(This is the Phase D foundation; see `docs/session-prompts/symbolic-extents-and-persistent-decode.md`.)

---

## The shape of the thing — why it's built this way

Look again at the load-bearing absence: a `Node` is `op + inputs + shape + dtype`, and
**nothing else**. No kernel. No backend. No execution plan.

That absence is the whole architecture in miniature:

- **The graph *is* the plan.** Because the node holds no backend and no kernel, the
  graph can carry the *entire* plan without the node knowing what runs it: the chosen
  backend lands in a `Graph` side-table (a stamp), alternative execution paths are arms
  of `Op::Branch` decision-point nodes *in the graph*, and the executor picks among them
  at runtime by live device load. There is no separate, persisted "plan" object in the
  destination — the optimizer writes its decisions *into the graph*, and the executor
  reads the graph.

- **`Op` vs `OpKind` is the same split, one level up.** `Op` is what a node *holds*
  (the IR variant). `OpKind` is what the *dispatcher keys on* to find a kernel. Keeping
  them distinct is what lets the IR stay pure description while dispatch is a separate
  lookup — the node never names a kernel.

- **`SymId` over pointers, ids over cells.** The node carries serializable ids so the
  base map can be written to disk and a fixed graph can be hashed and reused across
  passes. Same instinct as keeping the kernel off the node.

So when you read the code and find an `ExecutionPlan` threaded around, or a node's
kernel resolved through a static `OpKind`: those are **transitional scaffolding**, not
the destination. The destination is *graph carries the plan; registry carries the
kernels; executor decides at runtime*. Build new dispatch/fusion infrastructure into
the **graph + the runtime-mutable registry**, never into a side "plan." When the code
and the documented vision disagree, the vision wins.

---

## Where to go next

- [03-ir](architecture/03-ir.md) — the IR, the base map, the fused-op registry, layouts (the *why* behind Layer 2).
- [04-optimization](architecture/04-optimization.md) — decomposition/optimization maps, per-decision-point alternatives.
- [06-runtime](architecture/06-runtime.md) — the route picker and dispatch.
- [14-lifecycle](architecture/14-lifecycle.md) — the end-to-end flow + glossary (start here for the narrative).
