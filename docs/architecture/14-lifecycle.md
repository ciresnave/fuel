# Lifecycle: from model file to finished inference/training

**Status**: v0.6 (2026-06-25). v0.6 refines Stage 5's autoregressive-barrier story: the per-pass re-bind substrate (realize binds the runtime KV append offset / `cached_len` into one stable graph instead of baking it into a fresh per-token graph) has landed at the executor/session level, framed honestly as the Intended mechanism whose substrate has shipped but is not yet wired into production decode. Core claim unchanged.

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
  the graph"* redirection ([10-decisions-log](10-decisions-log.md)). **Phase A of that rebuild
  has now landed** (2026-06-15/16): the optimized form is an in-graph multi-path structure
  (`Op::Branch`), and `optimize_graph` is the **sole** realize-path optimizer — the separate
  `ExecutionPlan`/`PlanStore` dispatch path is deleted. **Phase B has also landed**: the
  optimizer now ranks on a per-path cost **vector**, retains a per-ending-device **Pareto
  frontier + crowding cap**, runs as a lock-step pathfinder/ranker/optimizer driver, and
  **compacts** the arena — so the bounded-frontier optimizer is now **Today**, not a gap.
  **Phase C has also landed**: the **runtime route picker** ("Picker 2") now chooses arms at
  `Op::Branch` points from live per-tier free memory (`pick_route`,
  `fuel-dispatch/src/ranker/route_picker.rs`) and realize follows the picked route (arm-0 when
  no branch is steered) — consulted per-branch, not per-node; and a **run-capture capability**
  exists on both GPU backends (CUDA graphs — `fuel-cuda-backend/src/capture.rs`; Vulkan
  reusable command buffers — `fuel-vulkan-backend/src/capture.rs`) for capture/replay/rebind of
  a run's launches. What still predates the redirection, and stays in the **Intended** column
  below, is: **load-time build** (the graph is still built inside `forward` and planned at
  realize time, a fresh graph per decode token — Phase D); **wiring run-capture into the
  executor** (the capability is built but not yet driven by dispatch — it amortizes only over
  repeated replay of the same run, which arrives with Phase D's persistent graph); **storage
  classes / sessions** (Phase D); and **mmap persistence** (Phase E). Each remaining gap is
  called out where it lands.

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
are the per-request work. **Today** the **plan** box has caught up — `optimize_graph` writes the
optimized multi-path form *in place* in the graph (Phase A) — but the **build graph** box has
not: the as-built code still builds the graph inside `forward` and runs `optimize_graph` at
realize time, rebuilding per request. That single remaining shift (load-time vs forward-time)
is now the largest Today-vs-Intended gap in this document, and it is what makes the
decode-per-token re-planning in stage 5 a gap rather than the design.

---

## Glossary (canonical terms)

**Graph** (a.k.a. *the DAG*, *the IR*) — fuel's intermediate representation: a directed
acyclic graph of operation **nodes**, held behind `Arc<RwLock<Graph>>`. Append-only;
nodes are immutable once created. *(Today: built lazily inside `forward`. Intended: built
once **at load** and **input-independent** — the graph is a property of the model, not of a
particular input; [03-ir](03-ir.md).)* (`fuel-graph`; `Graph` at `fuel-graph/src/lib.rs:1247`.)

**Node** — one operation: `{ op, inputs: Vec<NodeId>, shape, dtype }`. Identified by a
`NodeId` (a `usize` newtype). (`fuel-graph/src/lib.rs:1232`.)

**Op enum** — the closed set of ~80–90 primitive operations, plus `Op::Fused(FusedOpId, params)`
(delegates to the **fused-op registry**, frozen at startup) and `Op::Branch { reconverge_at }` —
the in-graph multi-path phi/merge **decision node**: the optimized form's divergent-then-
reconvergent routes live as real arena nodes (`fuel-graph/src/lib.rs:1006`, landed Phase A).
(`fuel-graph`; [03-ir](03-ir.md).)

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
friends. **Today** it runs `optimize_graph` over the graph (via the bridge's
`build_optimized_graph`), picks a route over the branches with the runtime selector (Phase C —
`pick_route`), and dispatches the **picked route** through
`PipelinedExecutor::realize_with_optimized_route` (arm-0 fallback for unsteered branches). Returns
host data (or leaves results device-resident). *(Intended: once the graph + plan are built at load,
realize reduces to "wait until the plan covers the requested roots, then dispatch"; today it
optimizes per realize.)*

**The plan** (a.k.a. *the optimized form* / *optimized DAG* in 03/04/06) — **Today: the plan IS
the graph.** `optimize_graph` (`fuel-dispatch/src/optimize.rs:192`) transforms the graph in place
into a multi-path structure whose decision points are `Op::Branch` nodes; that in-graph form is
the source of truth, and the dispatch order is read back from it by run extraction
(`lower_picked_route` for the runtime-picked route, `lower_runs_arm0` for the arm-0 fallback,
`fuel-graph/src/run.rs:328`). The `ExecutionPlan` (`fuel-dispatch/src/plan.rs:56`)
still exists but is **demoted** to a transient view `optimize_graph` returns — used only for
backend-stamping / residency / layout, never as the dispatch authority; `PlanStore` (its old
identity-keyed memoization) is **deleted**. *(**Today (Phase B):** the retained paths per device
are bounded by a **Pareto frontier + crowding cap** over a per-path cost **vector**
([04-optimization §Bounding the frontier](04-optimization.md)). The deliberate-fork pathfinder
(Phase A) is still the only path*finder*, so the frontier's breadth grows as more pathfinders land.)*

**Alternative set** — for one node, the ranked list of viable `(kernel, backend, device)`
choices. (`AlternativeSet`, `fuel-dispatch/src/ranker/alternative_set.rs`.) **Today** this is an
**internal placement detail** of `compile_plan` (which `optimize_graph` drives for per-node
cost/placement); it is no longer the realize-dispatch model. Realize dispatches the in-graph
`Op::Branch` decision points via the **picked-route lowering** (`lower_picked_route`; arm-0 for
any branch the runtime selector left unsteered), not a per-node top-N set. *(The deliberate-fork
pathfinder that turns a multi-placement choice into a
branch landed in Phase A; the per-device Pareto frontier + crowding cap that bounds the retained
candidates landed in Phase B — both are now Today.)*

**`compile_plan` / plan-time ranker ("Picker 1")** — per node it enumerates candidates, runs
the filter chain, computes cost, runs placement, and ranks. (`fuel-dispatch/src/plan.rs:488`.)
**Today** it is no longer the top-level optimizer: `optimize_graph` *drives it internally* for
per-node placement/cost, then emits `Op::Branch` decision points and returns the `ExecutionPlan`
as the transient stamping view.

**Runtime selector ("Picker 2", a.k.a. *route picker* in 06, *Router* in older
code/README)** — chooses among a branch's surviving paths using live telemetry. **Today (Phase
C) it is wired into the realize path**: `pick_route` (`fuel-dispatch/src/ranker/route_picker.rs`)
walks the `Op::Branch` points in topological order and, at each, builds an `AlternativeSet` over
the arms and consults the `ChainedSelector` (VramPressure → JudgeAware → Winner,
`fuel-dispatch/src/ranker/chained_selector.rs`) against live per-tier free memory, recording the
chosen arm. It is consulted **per branch** (a handful of times), not per node; realize lowers the
**picked route** (`lower_picked_route`) and falls back to **arm-0** for any branch left unsteered
(empty pick ⇒ the Phase B behavior). *"route picker," "selector," and "Router" all name this one
surface.*

**Judge** — an **offline** profiler that measures `(op, dtype, size_class, backend, device)`
latency/error and writes a profile the plan-time ranker reads. It does **not** measure ops
during normal dispatch. (`fuel-core/src/judge/`.)

**Work-item producer** *(today named the "compiler thread" in code —
`compiler_thread_body`, slated for rename)* — a worker thread *inside realize* that walks
the **picked-route dispatch order** (from `optimize_graph` + `pick_route` / `lower_picked_route`;
arm-0 where unsteered) and, for each node,
resolves the concrete kernel and binds its operands into a **WorkItem**, pushing them down a
channel. It does **not** compile machine code, and it is **not** a pipeline stage.
(`fuel-dispatch/src/pipelined.rs:861`.)

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
(optimization) → *optimized form* (the plan). **Today** the code builds the graph (in `forward`)
and then `optimize_graph` transforms it in place into the multi-path form (Phase A); the
fully-decomposed "base map" as a separately retained artifact is largely conceptual (the fused-op
registry and a first lowering rule exist; the broader rule engine is in progress per
[04-optimization](04-optimization.md):314+).

**Intended** ([03-ir](03-ir.md), 2026-06-14 redirection): the graph is **input-independent**
and built **at load**, not inside `forward`. Loading a native `.fuel` via `map_from_file`
reconstructs the whole graph (base map + storage + optimized paths) directly — no conversion;
importing a foreign checkpoint via `from_*` converts it **once** into the base map. Either way
the structure exists before any input does, and the per-request work in stages 4–6 binds inputs
to an already-built graph rather than building a new one. The as-built "no graph until `forward`
runs" path is the current simplification.

---

## Stage 3 — Plan (graph → its in-graph multi-path form)

**What it is:** decide *how* to run each node — which kernel, which backend, which device —
and in what order. The output is **the plan**, not execution.

**Today** the top-level optimizer is **`optimize_graph`** (`fuel-dispatch/src/optimize.rs:192`),
the sole realize-path optimizer (Phase A). It transforms the graph **in place**:

1. **Per-node placement/cost** — it drives `compile_plan` (`fuel-dispatch/src/plan.rs:488`)
   internally: for each kernel-bearing node, enumerate `(kernel, backend, device)` candidates →
   filter chain (`PrecisionFloor` hard, then `StridedInputPref` / `BitStablePref` soft) → cost
   (Layer-1 static `CostFn`, refined by Layer-2 **Judge** data, `cost.rs:155`) → carry-forward
   placement DP, ranked on the per-path **cost vector** (Pareto dominance, winner time-first),
   and retained per ending device by a **Pareto frontier + crowding cap** (`KEEP_PER_DEVICE`),
   not a fixed top-N. `optimize_graph` itself runs as a lock-step pathfinder/ranker/optimizer
   driver, and a standalone **compaction** pass can drop orphaned arena debris (Phase B).
2. **Emit branches** — where a node has ≥2 viable placements *and a single consumer*, the
   deliberate-fork pathfinder (`seed_placement_fork_branches`) records the choice as an
   `Op::Branch` decision point: arm-0 = the DP winner (the live route), arm-1 = the runner-up
   placement (an orphaned recording). Ordinary DAG fan-out is **not** flagged (that is the gate).
3. **Return a transient view** — `optimize_graph` returns the `ExecutionPlan` it built, used
   only for backend-stamping / residency / layout, never as the dispatch authority. The dispatch
   order is read back from the graph by **arm-0 run extraction** (`lower_runs_arm0`).

`PlanStore` (the old identity-keyed plan memoization) and the legacy `compile_plan`-drives-dispatch
path are **deleted**; `optimize_graph` runs once per realize.

**Today vs Intended — what Phase A left for later:**

- **Load-time planning is not wired.** `optimize_graph` runs *at realize time, inside the
  forward* (the graph is still built in `forward` — stage 2), not at model-load. Moving the
  build + optimize to load is the Phase D planner program ([06-runtime](06-runtime.md) intends it).
- **The bounded frontier is in (Phase B).** The optimizer ranks each path on a per-path cost
  **vector** (one central time metric + per-tier memory + discrete precision/accuracy, Pareto
  dominance), bounds retained paths per device by a **Pareto frontier + crowding cap**
  (`KEEP_PER_DEVICE`, retiring the fixed top-N), runs as a lock-step pathfinder/ranker/optimizer
  driver, and has a **compaction** pass ([04-optimization §Bounding the frontier](04-optimization.md);
  the prototype confirmed ~10² paths, lossless at keep ≈ 32/device). The remaining gap is
  *breadth* — Phase A's deliberate-fork seed is still the only path*finder*; fusion, algebraic,
  and dtype-lowering pathfinders come later.
- **Realize follows the runtime-picked route (Phase C, landed).** The route picker
  ("Picker 2") now chooses a *non-arm-0* arm at each branch from live per-tier free memory
  (`pick_route`); realize lowers the **picked route** (`lower_picked_route`) and falls back to
  **arm-0** for any branch left unsteered. (Through Phase B realize was arm-0-static.)
- **"Pre-resolved at plan time"** is only partly true — see stage 4: the binding-table lookup
  still happens during realize, in the work-item producer.

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
2. **Optimize + stamp:** run `optimize_graph` (the bridge's `build_optimized_graph`), which
   transforms the graph in place into its multi-path form and returns the transient
   `ExecutionPlan`; write each winner's backend into the graph's `target_backend` side-table from
   it, and stitch residency/contiguity fixups. (No `PlanStore`, no coverage-wait latch — Phase A
   deleted them; `optimize_graph` runs once per realize.)
3. **Pick the route + dispatch:** resolve the runtime route over the optimized graph
   (`pick_route` consulting the production selector against live per-tier free memory), then call
   `PipelinedExecutor::realize_with_optimized_route` with the picked route (or
   `realize_with_optimized` / arm-0 when nothing is steered, e.g. `FUEL_DISABLE_RUNTIME_SELECTOR=1`).

**The two threads** (`realize_inner`, `pipelined.rs:504`):

- **Work-item producer** (`compiler_thread_body`, `pipelined.rs:861`): holds one read lock
  on the graph, walks the **picked-route dispatch order** (`lower_picked_route`; arm-0 where no
  branch was steered), and for each node calls `resolve_compiled`, which resolves the kernel via
  the **binding-table lookup** (`compile_node`). The runtime selector (Picker 2) was consulted
  **once per branch** when the route was picked (stage 3, Phase C), not per node here. The
  resolved kernel + operands become a **WorkItem** pushed down a
  channel. (This is "compile" only in the sense of *lowering one node to a runnable unit* — hence
  the clearer name, work-item producer.)
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

**Today vs Intended:** binding-table kernel resolution happens **at realize time inside the
work-item producer** (`compile_node`), not at plan-build time — the consultation lives on the
producer thread, not the executor thread, but not all the way back to a load-time plan.

Runs **exist** in the graph (run extraction delimits them), and the **route picker** (Phase C)
already chooses a path only at the few **branch points** between runs (`pick_route`, consulted
per branch — not per node). Two pieces of the run-as-unit end state differ in status:

- **Picking over runs: done (Phase C).** Realize lowers the picked route and the selector fires
  once per branch, exactly the "pick only at branch points" intent.
- **Dispatching a run as a captured unit: capability built, wiring Intended (Phase D).** A
  run's launches can be captured into a *pre-recorded CUDA Graph* (`fuel-cuda-backend/src/capture.rs`)
  or a *reusable Vulkan command buffer* (`fuel-vulkan-backend/src/capture.rs`) and replayed /
  rebased onto new operands. But the executor still dispatches **per WorkItem** (per node): a
  captured graph only pays off when the *same* run is replayed many times, and the sole
  repeated-replay point (the decode loop) builds a fresh graph per token until Phase D gives runs
  a stable cross-realize identity. So Phase D wires capture into dispatch; the primitive is
  proven now.

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

**Why this matters (and a known gap):** because the decode graph is *fresh every token*,
**`optimize_graph` runs from scratch every decode token** — Phase A deleted `PlanStore`, so
there is no plan memoization at all. The decode-step graph is *structurally identical* step to
step (only the host scalar `cached_len` changes), so re-optimizing each token is pure waste. The
"1.8×/token" planning result that originally motivated memoization was measured on a synthetic
single-growing-graph loop and **does not reach this production decode path**.

Under the redirection this gap closes *by construction*: the decode graph is **not** rebuilt per token — the loaded, input-independent graph is reused across steps, and the only thing that advances is **session-class** storage (the KV-cache, keyed by `SessionId`) together with the runtime values (the write offset `cached_len`) **re-bound per pass** rather than baked into a fresh graph. The substrate for that re-bind has landed at the executor/session level: realize accepts a per-pass environment that binds the runtime value of each symbolic scalar (the KV append offset) into one stable graph, so a step *re-binds and re-runs* instead of re-building and re-planning. Production decode does not yet use it — it still bakes the host scalar and mints a fresh graph each token — so this is the **Intended** mechanism with its substrate proven, not the as-built decode path. Plan reuse then falls out of the graph being the same object, instead of needing a structural-hash plan cache bolted onto fresh-every-step graphs.

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
2. **"the plan"** is canonical for the **optimized form**, which **Today** IS the in-graph
   multi-path structure (`Op::Branch` decision points); the standalone `ExecutionPlan` is now
   only a transient stamping view, not "the plan."
3. **"runtime selector (Picker 2)"** is canonical for what 06-runtime calls the "route
   picker" and older code calls the "Router"; **"plan-time ranker (Picker 1)"** is the
   `compile_plan` ranking pass. Both are "pickers," at different times.
4. The **work-item producer and executor are inside realize**, not pipeline stages.
5. **"the plan is the graph"** — the optimized multi-path form lives *in* the graph as
   `Op::Branch` decision points, not in a separate artifact. **As of Phase A this is the as-built
   model**, not just Intended: `optimize_graph` writes it in place and the `ExecutionPlan` is
   demoted to a transient stamping view ([10-decisions-log](10-decisions-log.md) 2026-06-14;
   `docs/session-prompts/plan-is-graph-rebuild.md`).

---

## See also

- [03-ir](03-ir.md) — the graph, Op enum, fused-op registry, the three-artifact model.
- [04-optimization](04-optimization.md) — decomposition, the plan, placement, load-time planning.
- [05-backend-contract](05-backend-contract.md) — what backends advertise; the KernelRef ABI.
- [06-runtime](06-runtime.md) — the runtime selector, dispatch, the executor.
- [10-decisions-log](10-decisions-log.md) — the 2026-06-14 "plan is the graph" redirection that the **Intended** column above tracks.
- ROADMAP §"Post-wipe resume addendum" — the planner-program gaps (Stages 4b–6) referenced above.
