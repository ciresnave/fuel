# Identity

**Status**: v0.2 (draft, 2026-05-08).

What fuel is, what it isn't, and what makes it competitive.

---

## What fuel is

**Fuel is a DAG-first ML framework for Rust.**

Every model — whether loaded from a checkpoint, built programmatically, or assembled from layers — is represented as a directed acyclic graph of operations. The DAG is fuel's intermediate representation, the surface every component reads from and writes to, and the place every architectural decision is made:

- *What* a model computes lives in the DAG's structure (which ops, which dependencies).
- *How* the model runs — which backend, which device, which kernel variant, which fused replacement, which transfer paths between devices, which residency policies, which numerical-precision tradeoffs — is decided by the optimizer reading the DAG and the cost data backends report into it.
- *When* each step runs is decided by the runtime route picker reading the optimizer's annotated DAG and current backend telemetry.

This is the central architectural commitment: **the DAG is the source of truth for every decision, and the optimizer that operates on it is where intelligence about the model lives.**

## The five competitive edges

Most ML frameworks are organized around eager execution with optional graph capture. Backend-internal fusion (XLA, TVM, torch.compile) decides *at backend compile time* which kernels combine. This is enough to ship most models well — but it leaves five categories of optimization unreachable to backend-internal approaches alone. Fuel commits to all five from the start; together they form fuel's competitive position.

### 1. Cross-backend placement aware of fusion catalogs

A backend that fuses internally cannot decide "this fused matmul+bias+relu costs X on CUDA, but unfused on Vulkan it costs Y, and the bias is already produced on Vulkan, so unfused-on-Vulkan wins overall." Fuel's optimizer sees every backend's fusion catalog before placement decisions, including transfer costs between devices when crossing them is a candidate. Backend-side fusion is invisible to this kind of decision; fuel's framework-side fusion catalog is not.

### 2. Algebraic-equivalence rewrites that compress compute

`(a+b) * (a-b)` and `a² - b²` produce the same result with different op counts and different memory traffic. `matmul(A, B) + matmul(A, C)` and `matmul(A, B+C)` differ by a full matmul. Distributivity, associativity, factoring, identity-elimination, common-subexpression hoisting across non-trivial boundaries — these are real optimization wins that backend-internal fusion never sees because the patterns span ops the backend would never combine. Fuel's optimization layer searches for these rewrites by treating the DAG as algebraic expressions, not just dependency edges. Pattern matchers can be declarative (with variables) or callable functions; the optimizer is engine-agnostic.

### 3. Top-N route preservation for runtime adaptation

Most frameworks commit to a single optimal plan at compile time. Fuel's optimizer expands the DAG to keep multiple competitive routes — a **bounded per-device Pareto frontier** (capped by crowding distance, not a fixed global N) ranked by a cost vector. The runtime route picker reads current backend telemetry — memory pressure, queue depth, currently-loaded weights, request priority — and picks among preserved routes per-request. The same model serves a fast path under low load and a memory-conserving path under contention, without recompilation.

### 4. Pattern-harvest-driven fused-op development

When users opt in (off by default), fuel records the longest unfused op sequences and the most-frequently-repeated sequences in their workloads, anonymizes them, and reports them to the project's servers. This drives which fused kernels get developed next: prioritize by what real users would benefit from, not by what's familiar from prior frameworks. The structural property is that fuel knows what to fuse before competitors do, because it's measuring what real production workloads actually need fused. See [08-pattern-harvest](08-pattern-harvest.md) for privacy and protocol details.

### 5. Per-op tolerance budgets that unlock approximate optimizations

A user, a developer, or a model definition can specify how much numerical deviation from the strict-equivalence result is acceptable, per-op or per-region. With non-zero tolerance budgets, the optimizer reaches optimizations that strict-equivalence frameworks can't: mixed-precision lowering (F32→BF16 in regions where error allows), FP reassociation (`(a+b)+c → a+(b+c)`, mathematically equivalent for reals but not for IEEE floats), fast/approximate intrinsics (`__expf`, fast rsqrt, polynomial softmax), on-the-fly quantization, sparse approximations. The optimizer tracks cumulative error along each candidate route; routes that exceed budget are pruned. Most production ML frameworks treat precision as a binary or global mode; per-op tolerance budgets that the optimizer reasons about are uncommon. See [07-tolerance](07-tolerance.md).

## Two supporting commitments

The five edges depend on two structural commitments that distinguish fuel from contemporaries:

**Backends advertise; they don't decide.** Every backend (CPU, CUDA, Vulkan, Metal, AOCL, MKL, future ones) reports its kernels, its capabilities, its measured cost data, and its current telemetry to the optimizer. Backends never choose between alternatives at the strategic level — they execute what the optimizer hands them. Strategic decisions (placement, routing, fusion, dtype choice, tolerance budget consumption) live in fuel-core. Tactical decisions internal to a single kernel call (which CUDA tile shape, which BLAS variant for a chosen backend) may live in the backend, with cost feedback flowing up.

**Cost data is empirical, not predicted.** Static FLOP-counting cost models are rough and miss bandwidth saturation, kernel-launch latency, queue contention, hardware quirks. Fuel runs an empirical Judge that measures every (op, dtype, size_class) on every (backend, device) and produces a profile-driven dispatch surface. Static estimates registered with fused ops or rules are advisory; actual measurements override them. The runtime adapts as profile data accumulates. The same empirical mechanism evolves to measure error contribution per op (not just latency), feeding tolerance-aware routing.

## What fuel isn't

The identity is sharpened by what fuel deliberately rejects. Full list in [09-non-goals](09-non-goals.md); the headline rejections:

- **Lazy-only execution; no eager mode.** Eager hides the DAG and prevents optimization. Fuel's only execution path is lazy + explicit `.realize()`. The performance target is **external**: because the optimizer picks the best available implementation of every op and re-adapts to live device state, lazy-realize should keep up with or outperform *every* eager framework — with Candle, fuel's near-unchanged fork parent, as the first yardstick (fast enough that Candle's eager looks slow by comparison) and llama.cpp / PyTorch / ONNX Runtime as the bar beyond it. Since the in-repo eager path was retired (Phase 7.5), this commitment is enforced by an out-of-repo benchmark gate (see ROADMAP §Benchmarking), proven with measurements before any non-alpha release — not "eager is forbidden because we say so." JAX has demonstrated this idiom is learnable at scale.
- **Not backend-internal-fusion.** Backends don't choose what to fuse; the optimizer does. Backend-internal fusion would be invisible to placement decisions, to algebraic rewrites, and to tolerance-budget reasoning.
- **Not framework-agnostic dispatch.** Fuel doesn't try to be a generic DAG executor that works with arbitrary user-defined ops at runtime. The Op enum is closed and small; fused ops are extensible through a frozen-at-startup registry, not at runtime.
- **Not multi-dialect IR.** Two layers (primitive Op + fused-op registry) is enough. MLIR-style multi-dialect frameworks pay for expressiveness fuel doesn't need.
- **Not a Python interop layer.** Fuel is Rust-native. Python bindings, if they happen, are a leaf concern at the orchestration layer, not architectural.

## How this identity is enforced

Architecture documents are easy to write and easy to ignore. The identity is durable only if every phase, every PR, every design decision is checked against it.

**The check**: a change passes if it makes one of these statements *more* true and none *less* true:

1. *More decisions are visible to and made by the DAG-level optimizer.*
2. *More cost information flows from backends to the optimizer.*
3. *More algebraic-equivalence rewrites are reachable by the optimization machinery.*
4. *More of the optimization space is reachable under tolerance budgets without compromising strict-equivalence default behavior.*

A change fails if it makes any of those *less* true — e.g. moves a decision into a backend, hides cost data behind a backend abstraction, restricts pattern matching to literal sequences when algebraic patterns are needed, or couples the strict path to tolerance-aware machinery in a way that complicates strict-equivalence usage.

Borderline cases get discussed against this section before they ship. The check evolves as the architecture evolves; revisions are recorded in [10-decisions-log](10-decisions-log.md).

## The bet, stated plainly

Fuel bets that the next decade's ML framework wins compound across two layers, not one:

- **Bigger and better fused kernels matter, and will keep mattering.** Hand-tuned fused kernels (FlashAttention, fused linear+activation, FlashDecoding for inference) are real wins that fuel will continue to ship in its registry. Backends will keep getting faster kernels for the patterns that matter.
- **Above the kernel layer, optimization techniques that span ops, span backends, adapt at runtime, and trade controlled error for compute will keep finding wins backend-internal fusion can't reach.** This is where most ML frameworks today stop investing seriously. Fuel doesn't.

The bet is that *both layers compound*: a great fused kernel is more valuable in fuel because the optimizer can place it across backends, swap it in via algebraic rewrite, fall back to a top-N alternative under load, and know when a tolerance-relaxed variant suffices. Frameworks that ship only the kernel layer see linear gains as kernels improve; fuel sees superlinear gains as both layers improve together.

If the bet is wrong — if backend-internal fusion plus eager execution is enough and the upper-layer optimizations don't move the needle — then fuel is over-engineered. The bet is that the leverage is real, that the leverage compounds, and that other frameworks will need years to retrofit what fuel ships from the start.

---

## See also

- [03-ir](03-ir.md) — the DAG, the Op enum, the fused-op registry, layouts.
- [04-optimization](04-optimization.md) — DecompositionMap, OptimizationMap, top-N expansion, sliding window, declarative + callable rule engines.
- [05-backend-contract](05-backend-contract.md) — what backends provide and what they don't decide.
- [07-tolerance](07-tolerance.md) — per-op error budgets, hierarchical specification, what tolerance unlocks.
- [08-pattern-harvest](08-pattern-harvest.md) — opt-in telemetry that drives fused-op prioritization.
- [09-non-goals](09-non-goals.md) — full list of what fuel deliberately doesn't try to be.
- ROADMAP §"Identity" — the layered-Rust-ML-framework framing; this document refines that with the DAG-first commitment and the five competitive edges.
