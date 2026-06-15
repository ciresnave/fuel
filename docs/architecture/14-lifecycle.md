# Lifecycle: from model file to finished inference/training

**Status**: v0.2 (2026-06-14).

This is the one document that walks the **whole path**, in order: from "load a model
from disk" to "inference or training has finished." Every other architecture section
covers one slice by concern ([03-ir](03-ir.md) the IR, [04-optimization](04-optimization.md)
the plan, [06-runtime](06-runtime.md) execution); this one stitches them into a single
narrative so users, developers, and agents share one mental model and one vocabulary.

Two reading rules:

- **Terminology is load-bearing.** The glossary below defines each term *once*. Where the
  codebase or older docs use two names for one thing, the canonical name is given and the
  synonym is footnoted. Use the canonical names.
- **Today vs Intended.** Fuel's architecture is partly aspirational. Each stage marks what
  the code does **Today** versus what the architecture **Intends**. Do not read the
  intended steady-state as if it were all implemented — that mismatch is exactly what this
  document exists to prevent. The **Intended** column reflects the 2026-06-14 *"the plan is
  the graph"* redirection ([10-decisions-log](10-decisions-log.md)); the as-built code largely
  predates it — a separate `ExecutionPlan`, per-node top-N alternatives, a fresh graph per
  token, and planning at realize time — so each such gap is called out where it lands.

---

## The pipeline in one line

```text
load → build graph → plan → realize → (inference loop | training loop)
```

That is the whole thing. Everything else is detail *inside* one of those five boxes. In
particular, the **"work-item producer"** and the **"executor"** are two threads *inside the
realize box* — they are not stages of the pipeline. (This is the single most common
confusion; see the glossary and stage 4.)

Where the boundary sits matters. Under the redirection, the first two boxes — **build graph**
and **plan** — are *load-time and input-independent*: the graph, with its optimized multi-path
form, is built once when the model loads, not rebuilt per request. **realize** and the loops
are the per-request work. The as-built code instead builds the graph inside `forward` and plans
at realize time; that single shift (load-time vs forward-time) is the largest Today-vs-Intended
gap in this document, and it is what makes the decode-per-token re-planning in stage 5 a gap
rather than the design.

---

## Glossary (canonical terms)

**Graph** (a.k.a. *the DAG*, *the IR*) — fuel's intermediate representation: a directed
acyclic graph of operation **nodes**, held behind `Arc<RwLock<Graph>>`. Append-only;
nodes are immutable once created. *(Today: built lazily inside `forward`. Intended: built
once **at load** and **input-independent** — the graph is a property of the model, not of a
particular input; [03-ir](03-ir.md).)* (`fuel-graph`; `Graph` at `fuel-graph/src/lib.rs:1247`.)

**Node** — one operation: `{ op, inputs: Vec<NodeId>, shape, dtype }`. Identified by a
`NodeId` (a `usize` newtype). (`fuel-graph/src/lib.rs:1232`.)

**Op enum** — the closed set of ~80–90 primitive operations, plus a single open arm
`Op::Fused(FusedOpId, params)` that delegates to the **fused-op registry** (frozen at
startup). (`fuel-graph`; [03-ir](03-ir.md).)

**LazyTensor / Tensor** — the handle a user (or model code) holds. Calling `.matmul()`,
`.softmax()`, etc. on it appends nodes to the graph and returns a new handle. It carries
no data — only a reference to a node. (`fuel-core/src/lazy.rs`, `fuel-graph` `Tensor`.)

**Storage** — an actual typed, contiguous memory buffer (host or device), held as
`Arc<RwLock<Storage>>`. Lives in the `fuel-memory` crate. The graph references storage by
node; weights and KV-caches are storage buffers shared across realizes via their `Arc`.
*(Renamed from `fuel-storage` 2026-06-13.)* *(Intended: each buffer carries a **storage
class** — **shared** (weights, `Op::Const`; one copy across all sessions), **session**
(KV-cache and anything keyed by a `SessionId`), or **transient** (activations; freed when
dead). Today the class is implicit in how each `Arc` is held; [03-ir §Storage classes and
sessions](03-ir.md).)*

**realize** — the call that crosses from graph-building to execution: `.realize_f32()` and
friends. Under the load-time-planning design it is conceptually "wait until the plan covers
the requested roots, then dispatch." Returns host data (or leaves results device-resident).
(`PipelinedExecutor::realize*`, `fuel-dispatch/src/pipelined.rs:406+`.)

**The plan** (a.k.a. *the optimized form* / *optimized DAG* in 03/04/06) — the
`ExecutionPlan`: a topological `order`, a sparse map of per-node **alternative sets**, and a
`generation` stamp. Built by `compile_plan`. (`fuel-dispatch/src/plan.rs:56`.) This document
uses **"the plan"** as canonical; "optimized form" is the same thing in conceptual prose.
*(Intended: the optimized paths are not a separate artifact at all — they live **in the
graph** as a bounded multi-path structure, so "the plan **is** the graph." The standalone
`ExecutionPlan` is the as-built realization of it; [03-ir §The optimized form](03-ir.md),
[04-optimization](04-optimization.md).)*

**Alternative set** — for one node, the ranked list of viable `(kernel, backend, device)`
choices (default top-N = 3). The plan stores a set per kernel-bearing node, not N whole
competing graphs. (`AlternativeSet`, `fuel-dispatch/src/ranker/alternative_set.rs`.)
*(Intended: choices attach to **decision (branch) points**, not to every node, and the set
of retained paths per device is bounded by a **Pareto frontier + crowding cap** rather than a
fixed top-N; [04-optimization §Bounding the frontier](04-optimization.md). The fixed default
top-N = 3 is the as-built shape.)*

**`compile_plan` / plan-time ranker ("Picker 1")** — the optimizer pass that builds the
plan: per node it enumerates candidates, runs the filter chain, computes cost, runs
placement, and ranks. (`fuel-dispatch/src/plan.rs:488`.)

**Runtime selector ("Picker 2", a.k.a. *route picker* in 06, *Router* in older
code/README)** — at realize time, chooses among a node's stored alternatives using live
telemetry. The default is `ChainedSelector` (VramPressure → JudgeAware → Winner).
(`fuel-dispatch/src/ranker/chained_selector.rs`.) *"route picker," "selector," and "Router"
all name this one surface.*

**Judge** — an **offline** profiler that measures `(op, dtype, size_class, backend, device)`
latency/error and writes a profile the plan-time ranker reads. It does **not** measure ops
during normal dispatch. (`fuel-core/src/judge/`.)

**Work-item producer** *(today named the "compiler thread" in code —
`compiler_thread_body`, slated for rename)* — a worker thread *inside realize* that walks
the plan's order and, for each node, resolves the concrete kernel and binds its operands
into a **WorkItem**, pushing them down a channel. It does **not** compile machine code, and
it is **not** a pipeline stage. (`fuel-dispatch/src/pipelined.rs:734`.)

**Executor** — the *calling* thread's loop that consumes WorkItems and runs them: gather
inputs, allocate output, invoke the kernel, evict dead buffers. Runs concurrently with the
work-item producer purely to pipeline. (`fuel-dispatch/src/pipelined.rs:521`.)

**WorkItem** — one unit of executable work: a node, its inputs, dtype/shape, a
`target_backend`, a `WorkItemKind`, and a resolved kernel. (`fuel-dispatch/src/pipelined.rs:264`.)

**KernelRef** — the kernel ABI: a Rust function pointer
`fn(&[Arc<RwLock<Storage>>], &mut [...], &[Layout], &OpParams) -> Result<()>`. The kernel is
invoked synchronously. (`fuel-dispatch/src/kernel.rs:152`.)

**Backend / device** — a backend (CPU, CUDA, Vulkan, Metal) *advertises* its kernels,
capabilities, and telemetry; it never decides placement or fusion. A `DeviceLocation` names
a concrete device. (`fuel-core-types`, [05-backend-contract](05-backend-contract.md).)

**dispatch-chunk boundary** — the point between two consecutive WorkItems whose
`target_backend` differs; the unit at which the runtime re-checks the topology generation.
(`fuel-dispatch/src/pipelined.rs:524`.)

---

## Stage 1 — Load (model file → weights in memory)

**What it is:** read a checkpoint from disk (or the HF Hub) and get its weights into memory.

**Today** (`LlamaModel::from_hub`, `fuel-core/src/lazy.rs:6263` → `load_from_mmapped`,
`:5515`): hf-hub fetches `config.json` and the safetensors shards; the shards are mmap'd
(`MmapedSafetensors`), then each tensor is **eagerly copied** into an owned `Arc` buffer,
upcasting F16/F64 to F32 and transposing HF `out×in` weight layout to fuel's `in×out`. The
mmap is then dropped. Weights live as owned host `Arc` buffers (`WeightStorage`,
`lazy.rs:4378`) — **not** as graph nodes, and **not** mmap-resident.

**Intended** ([04-optimization](04-optimization.md) §load-time planning, [11-persistence](11-persistence.md)):
weights stay mmap-resident and are paged in *on the schedule the plan implies* (the plan
doubles as a prefetch schedule), so time-to-first-token isn't gated on copying the whole
model. The eager-copy path is the current simplification.

---

## Stage 2 — Build the graph (weights + model code → a DAG)

**What it is:** turn "run this model on this input" into a graph of operations, without
computing anything yet.

**Today:** there is **no graph until the model's `forward` runs.** The first
`LazyTensor::from_f32(...)` call mints a fresh `Graph` (`Arc::new(RwLock::new(Graph::new()))`,
`fuel-graph/src/lib.rs:2260`). As the forward proceeds, each op (`embed.index_select`,
`matmul`, `rope`, `softmax`, …) appends a node; weights are re-emitted as `Op::Const` nodes
that reference their storage buffer (`apply_linear`, `lazy.rs:4502`). The result is the
graph for *this* forward — what the architecture calls the **user-facing form** (a mix of
primitive and fused nodes). A view op (reshape/transpose/slice) is metadata-only via the
layout side-table — zero-copy.

**How this maps to the conceptual model:** [03-ir](03-ir.md) frames three artifacts —
*user-facing form* → (decomposition) → *base map* (primitive-only, canonical) →
(optimization) → *optimized form* (the plan). **Today** the code builds the graph and then
`compile_plan` annotates it; the fully-decomposed "base map" as a separately retained
artifact is largely conceptual (the fused-op registry and a first lowering rule exist; the
broader rule engine is in progress per [04-optimization](04-optimization.md):314+).

**Intended** ([03-ir](03-ir.md), 2026-06-14 redirection): the graph is **input-independent**
and built **at load**, not inside `forward`. Loading a native `.fuel` via `map_from_file`
reconstructs the whole graph (base map + storage + optimized paths) directly — no conversion;
importing a foreign checkpoint via `from_*` converts it **once** into the base map. Either way
the structure exists before any input does, and the per-request work in stages 4–6 binds inputs
to an already-built graph rather than building a new one. The as-built "no graph until `forward`
runs" path is the current simplification.

---

## Stage 3 — Plan (graph → an ExecutionPlan)

**What it is:** decide *how* to run each node — which kernel, which backend, which device —
and in what order. The output is **the plan**, not execution.

**Today** (`compile_plan`, `fuel-dispatch/src/plan.rs:488`): for each kernel-bearing node,
in order:

1. **Enumerate** candidate `(kernel, backend, device)` choices (`enumerate_candidates`).
2. **Filter** them through the chain — `PrecisionFloor` (hard: drop choices that violate the
   node's precision requirement), then soft preferences (`StridedInputPref`, `BitStablePref`).
3. **Cost** each survivor: Layer-1 static `CostFn` (a pessimistic estimate), refined by
   Layer-2 **Judge** data where the Judge has profiled that cell (`compute_static_costs`,
   `cost.rs:155`).
4. **Place + rank**: the carry-forward placement DP (or a greedy fallback) accounts for
   cross-device transfer cost, then ranks by `composite_ns` and keeps the top-N as the
   node's **alternative set**, with a winner.

The plan (`order` + per-node alternative sets + `generation`) is memoized in the
**PlanStore**, keyed by graph identity + device (`plan_store.rs`). This pass is **Picker 1**
(the plan-time ranker).

**Today vs Intended — three real gaps to know:**

- **Load-time planning is not wired.** The only `Planner::warm` call runs *inside the
  forward, just before realize* (`lazy.rs:5295`), not at model-load. So planning happens at
  realize time, per forward. ([06-runtime](06-runtime.md) v1.1 intends planning to start at
  load; that is the unfinished planner program, Stages 4b–6.)
- **The intended optimizer is `optimize_graph`, run at load** — pathfinders propose alternative
  paths, rankers score each on a per-path cost *vector* (a single central time metric + per-tier
  memory + discrete precision/accuracy), optimizers rewrite, and the retained paths per device
  are bounded by a **Pareto frontier + crowding cap** ([04-optimization §Bounding the
  frontier](04-optimization.md)). The as-built `compile_plan` is instead per-node, fixed top-N,
  synchronous/single-pass, and runs at realize time. The redirection ([10-decisions-log](10-decisions-log.md)
  2026-06-14) makes "decisions at branch points, paths in the graph" the target; the
  frontier-pruning prototype confirmed it stays bounded (~10² paths, lossless at keep ≈ 32/device).
- **"Pre-resolved at plan time"** is only partly true — see stage 4: the binding-table
  lookup actually happens during realize, in the work-item producer.

---

## Stage 4 — Realize (the plan → results)

**What it is:** the boundary where the graph stops being IR and actually runs. This is the
box that contains the work-item producer and the executor.

**Entry** (`fuel-core`): `LazyTensor::realize_f32()` (`lazy.rs:1308`) → the **bridge**
(`pipelined_bridge.rs`). The bridge does the prep that execution needs:

1. **Prep:** splice an `Op::Copy { target: Cpu }` node at each realize root (so the
   device→host download is itself a graph node), and upload every reachable `Op::Const` into
   a **StorageCache** (`HashMap<NodeId, Arc<RwLock<Storage>>>`). Long-lived buffers (weights,
   KV-cache) are seeded here so they are never re-uploaded.
2. **Plan + stamp:** get/extend the plan from the PlanStore (this is the **coverage-wait** —
   ensure the plan covers every node the roots need; if another thread is planning the same
   graph/device, wait on its latch), then write each winner's backend into the graph's
   `target_backend` side-table and stitch residency/contiguity fixups.
3. Call `PipelinedExecutor::realize_with_plan_and_selector`.

**The two threads** (`realize_inner`, `pipelined.rs:454`):

- **Work-item producer** (`compiler_thread_body`, `pipelined.rs:734`): holds one read lock
  on the graph, walks the plan's `order`, and for each node calls `resolve_compiled` — if the
  plan carries an alternative set, the **runtime selector (Picker 2)** chooses among the
  top-N using live telemetry; otherwise the static winner is taken; otherwise it falls back
  to a live **binding-table lookup**. The resolved kernel + operands become a **WorkItem**
  pushed down a channel. (This is "compile" only in the sense of *lowering one planned node
  to a runnable unit* — hence the clearer name, work-item producer.)
- **Executor** (the `for item in rx` loop, `pipelined.rs:521`, on the *calling* thread):
  for each WorkItem — (a) at a **dispatch-chunk boundary** (target_backend change), re-check
  the topology `generation` against the plan's stamp; a mismatch raises
  `Error::TopologyChanged`, which the bridge catches and retries with a rebuilt plan; (b)
  `execute_work_item` gathers input `Arc`s from the StorageCache, auto-contiguizes any
  non-contiguous input unless the kernel advertises `strided_input`, allocates the output on
  the target device, and invokes the `KernelRef`; (c) if the op is destructive, **evict** the
  consumed input from the cache (its `Arc` drop frees the device memory) unless it's a realize
  target.

The two threads run **concurrently purely to pipeline**: the producer prepares WorkItem N+1
while the executor runs WorkItem N. Run on one thread, the behavior would be identical, only
slower — which is why neither is a pipeline stage.

**Finish:** when the channel closes, the producer thread is joined; the target's storage is
read from the cache; because of the spliced `Op::Copy`, the result is CPU-resident and is
reinterpreted into a typed `Vec`. `realize_many` is the multi-root sibling (one shared plan
over all targets' dependency sets; shared subgraphs computed once).

**Today vs Intended:** [03-ir](03-ir.md):129 / the audit docs say "the executor calls
pre-resolved KernelRefs and never looks up kernels at execution time." In the real code the
binding-table lookup happens **at realize time inside the work-item producer**, not at
plan-build time. The plan path is *mostly* pre-resolved (alternative sets carry kernel
pointers), but plan-absent nodes and the fallback still do a live lookup. The consultation
moved off the executor thread into the producer thread — not all the way back to plan time.

Also intended (redirection): the unit of dispatch is a **run** — a fixed op-sequence between
two decision points — handed to the executor as a unit, with the **route picker** choosing a
path only at the **branch points** between runs, not at every op. The as-built executor
dispatches per WorkItem (per node) and the selector may fire at any kernel-bearing node;
collapsing straight-line spans into runs (so the picker is consulted a handful of times, not
once per node) is the intended optimization.

---

## Stage 5 — The inference loop (realize, repeatedly, autoregressively)

**What it is:** generate tokens one at a time, each depending on the last.

**Today** (`generate` → `generate_streaming_with_kv_context`, `lazy.rs:5749/5802`):

1. Allocate a fixed-capacity **KV-cache**: per layer, `[1, n_kv_heads, max_seq_len,
   head_dim]` zero buffers (`Op::Alloc` + `Op::ZeroFill`), held as `Arc`s **outside any
   graph** (`inference_context.rs:181`). `max_seq_len = prompt + max_new`.
2. **Prefill:** one forward over the whole prompt → the last position's logits.
3. **Decode loop:** `sample_logits` picks the next token **on the host** (argmax or
   sampled softmax) — this is the **autoregressive barrier**: step N+1's graph cannot be
   built until step N's logits are realized to the host and a token chosen. The chosen
   token is fed as the *single* input to the next forward.

Each `forward_with_kv_context_impl` (`lazy.rs:5168`) builds a **fresh graph** (new `Arc`),
binds the KV-cache buffers to fresh `Op::Const` placeholders, and at each layer writes that
step's K/V slab into the pre-allocated buffer via `Op::WriteSlice` at the host-scalar range
`(cached_len, cached_len + seq)`. It **realizes once** (`lazy.rs:5304`), then the graph is
dropped; the KV-cache `Arc`s survive because the cache owns them. `cached_len += seq`.

**Why this matters (and a known gap):** because the decode graph is *fresh every token* and
the PlanStore keys on graph identity, **every decode token currently re-plans from scratch**;
the per-step `warm` only helps that same step. The decode-step graph is *structurally
identical* step to step (only the host scalar `cached_len` changes), so the intended fix
(planner Stage 5, structural-hash plan memoization) is to reuse the plan across steps — *not
yet built*. The headline "1.8×/token" planning result was measured on a synthetic
single-growing-graph loop and **does not reach this production decode path** yet.

Under the redirection this gap closes *by construction*: the decode graph is **not** rebuilt
per token — the loaded, input-independent graph is reused across steps, and the only thing that
advances is **session-class** storage (the KV-cache, keyed by `SessionId`). Plan reuse then
falls out of the graph being the same object, instead of needing a structural-hash plan cache
bolted onto fresh-every-step graphs.

---

## Stage 6 — The training loop (forward → backward → step, repeatedly)

**What it is:** the same build-graph-then-realize loop, with a backward pass and an
optimizer step.

**Today** (`TrainState::step`, `fuel-core/src/train.rs:386`):

1. Build a **fresh graph**; bind each parameter's current storage `Arc` to an `Op::Const`
   placeholder (the same persistent-`Arc` pattern as the KV-cache).
2. The user closure builds the forward graph and a scalar `loss`.
3. `loss.backward()` (`fuel-graph/src/lib.rs:5608`) is **autograd as a graph rewrite**: it
   walks the graph in reverse-topological order and, per node, emits *new gradient nodes onto
   the same graph* — via the `GradientRule` registry (`grad.rs`) for migrated ops
   (Add/Mul/Relu/Where/comparisons) and an inline match for the rest (MatMul etc.). Fan-out
   gradients are summed with `Op::Add` (`accumulate_grad`). The result is a map from each
   forward node to its gradient node.
4. Append the optimizer update ops (SGD `w − lr·g`, or AdamW moment-update + bias-correction
   + decoupled weight decay), binding the current moment buffers as Consts.
5. **One realize** per step over `roots = [loss, new_params…, new_moments…]`
   (`realize_split`): the loss comes back to the host; updated parameters and optimizer state
   stay device-resident and are carried into the next step as `Arc`s.

**Today vs Intended:** training is single-device, F32-only, and allocates a fresh buffer per
updated parameter each step (no in-place optimizer update yet). Autograd is a forward-compatible
split — only some ops use the `GradientRule` registry; the rest still use the inline backward
match.

---

## The terminology decisions this document locks

1. **"work-item producer"** replaces "compiler thread" (the code symbol `compiler_thread_body`
   should be renamed to match; `compile_plan` keeps its name, leaving exactly one "compile" in
   the vocabulary — the plan build).
2. **"the plan"** is canonical for the `ExecutionPlan` / "optimized form."
3. **"runtime selector (Picker 2)"** is canonical for what 06-runtime calls the "route
   picker" and older code calls the "Router"; **"plan-time ranker (Picker 1)"** is the
   `compile_plan` ranking pass. Both are "pickers," at different times.
4. The **work-item producer and executor are inside realize**, not pipeline stages.
5. **"the plan is the graph"** — the optimized multi-path form lives *in* the graph, not in a
   separate artifact; "the plan" names that embedded structure in conceptual prose, while the
   standalone `ExecutionPlan` is the as-built implementation of it (item 2). This is the
   Intended model the stages above measure Today against ([10-decisions-log](10-decisions-log.md)
   2026-06-14).

---

## See also

- [03-ir](03-ir.md) — the graph, Op enum, fused-op registry, the three-artifact model.
- [04-optimization](04-optimization.md) — decomposition, the plan, placement, load-time planning.
- [05-backend-contract](05-backend-contract.md) — what backends advertise; the KernelRef ABI.
- [06-runtime](06-runtime.md) — the runtime selector, dispatch, the executor.
- [10-decisions-log](10-decisions-log.md) — the 2026-06-14 "plan is the graph" redirection that the **Intended** column above tracks.
- ROADMAP §"Post-wipe resume addendum" — the planner-program gaps (Stages 4b–6) referenced above.
