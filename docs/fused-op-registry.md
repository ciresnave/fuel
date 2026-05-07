# FusedOpRegistry: Open Registry for Fused Ops, Closed Enum for Primitives

**Status**: design v1, 2026-05-07. Iterating before code lands.

## TL;DR

Today's `Op` enum is a hybrid: ~60 variants mixing primitives (Add, Mul, Exp, MatMul) with fused abstractions (SoftmaxLastDim, RmsNormLastDim, FlashAttn, FusedLinear). The hybrid works, but it doesn't scale: every fused kernel a backend wants to register requires a new `Op` variant + executor arms in every backend + autograd entries + tooling support. The Op enum becomes the bottleneck for fusion.

**The split**: `Op` becomes a closed enum of *primitives only* (~80 variants, stablehlo-sized). Fused ops live in a separate `FusedOpRegistry` — an open, dynamically-populated registry indexed by `FusedOpId`. Each registry entry carries the fused op's primitive-subgraph signature (for fusion pattern recognition), its decomposition (for lowering), per-backend kernel implementations with cost estimates, and its backward op (anchored as another FusedOpId or a primitive subgraph). Graph nodes hold either `Primitive(Op)` or `Fused(FusedOpId)`.

**Why now**: the Phase 4 cost-based scheduler needs cross-backend fusion visibility *before* placement decisions to compare "matmul+bias+relu costs X on CUDA fused, Y on Vulkan unfused." Backend-side fusion (XLA's model) doesn't satisfy this — fuel's multi-backend Router needs every backend's fusion catalog visible to the pre-placement optimizer. A registry is the natural shape for that catalog.

**This integrates with PR 3's rule registry**, doesn't replace it. The FusedOpRegistry is the *source of truth* for fused ops; the RuleRegistry's lowering and fusion rules are *auto-derived* from each FusedOpEntry's decomposition + pattern. Today's hand-written PR 3 rules become declarative entries.

---

## Goals

- **Open registry, closed primitive enum.** `Op` stays small, exhaustively matched, panic-free. Fused ops are added without touching `Op`.
- **Cross-backend fusion visibility.** Every backend's fused-kernel catalog is visible to the pre-placement optimizer for cost-based device selection.
- **Bidirectional pattern↔fused-op mapping.** Lowering: FusedOpId → primitive subgraph. Fusion: primitive subgraph pattern → FusedOpId. Same registry, two indices.
- **One source of truth per fused op.** Each entry defines its decomposition, pattern, backend impls, and backward — no risk of drift between Op variant docs and lowering rule definitions.
- **Backend extensibility.** A backend adds a fused kernel by registering against a FusedOpId. No `Op` enum edit, no autograd edit, no executor arm edit.
- **No production panics.** Per project rule. Registry lookups validated at registration time; runtime dispatch has no missing-key paths.

## Non-goals (this work item)

- **Cost-based scheduler implementation.** This refactor *enables* it; the actual scheduler that consumes per-backend cost estimates is a separate phase.
- **e-graph / equality-saturation pattern matching.** PR 3's rule-driver style (anchored structural matching) is sufficient for the registry's pattern-recognition needs. e-graphs are a future optimization if pattern proliferation justifies them.
- **Multi-level dialect IR (MLIR-style).** The registry plus the `Primitive(Op) | Fused(FusedOpId)` node kind covers the practical need without committing to a full multi-dialect framework.
- **Backend-specific autotuning.** Cost estimates come from per-backend functions in registry entries; how a backend computes its estimate (static, profile-driven, autotuner) is a backend concern.

---

## Current state

### `Op` enum (fuel-graph/src/lib.rs)

~60 variants, mixed:

- **Primitives**: Add, Sub, Mul, Div, Neg, Sqr, Sqrt, Exp, Log, Sin, Cos, Tanh, Sigmoid, Silu, Gelu, Relu, Step, MatMul, Maximum, Minimum, AddScalar, MulScalar, PowI, Clamp, Cast.
- **Shape/view**: Transpose, Permute, BroadcastTo, Reshape, Slice, Concat, Unsqueeze.
- **Reductions**: SumAll, MaxAll, MinAll, MeanAll, SumDim, MaxDim, MinDim, MeanDim, ArgMaxDim, ArgMinDim, ReduceSumTo, ReduceMaxTo.
- **Fused / high-level**: SoftmaxLastDim, LayerNormLastDim, RmsNormLastDim, Rope, FusedLinear, Conv2D, ConvTranspose2D, FlashAttn, PagedAttn, QMatMul.
- **Backward helpers (fused)**: SoftmaxLastDimBackward, LayerNormLastDimBackward, RmsNormLastDimBackward, ReduceMaxToBackward.
- **Indexing**: IndexSelect, Gather, IndexAdd, ScatterAdd.
- **Memory/control**: Const, Copy, Move, Release.

Of these, ~25 are clearly fused/high-level (SoftmaxLastDim, RmsNormLastDim, Rope, Conv2D, ConvTranspose2D, FlashAttn, PagedAttn, FusedLinear, QMatMul, the 4 backward-helpers, etc.). These are the ones that move to the registry.

### `Node` (fuel-graph)

```rust
pub struct Node {
    pub op: Op,
    pub inputs: Vec<NodeId>,
    pub shape: Shape,
    pub dtype: DType,
}
```

`op: Op` becomes `kind: NodeKind` post-refactor.

### Existing fusion machinery

PR 3 (commit 3d7ca325) shipped `RuleRegistry` with `Rule` trait, `RuleFamily::{Lowering, Fusion}`, and a fixpoint driver. Today it has one rule pair: `SoftmaxLastDimLowerRule` + `SoftmaxLastDimFuseRule`, hand-written. The fuse rule walks back from a Div node and structurally matches the canonical 7-node subgraph; the lower rule emits that subgraph from `Op::SoftmaxLastDim`.

The pattern that would generalize: every fused op's `(decomposition, pattern, backend_kernels)` becomes a registry entry; lowering and fusion rules are auto-generated from the entry.

### Backend registration today

`fuel-storage::dispatch.rs` registers kernels against `OpKind` (mirrored from `Op`). Each `(OpKind, dtype-tuple, backend)` triple resolves to a wrapper function. New fused ops require new `OpKind` variants + register calls in dispatch.rs + (sometimes) executor arms in fuel-graph-executor.

This works but spreads the "what is this fused op?" knowledge across `Op`, `OpKind`, `OpParams`, dispatch wrappers, executor arms, autograd rules, op_short_name, op_key, and now PR 3 lowering/fusion rules. The FusedOpRegistry centralizes it.

---

## Proposed architecture

### `Op` becomes primitive-only

Closed enum of primitives + memory/control. Target ~85 variants:

- All primitives currently in `Op` stay (Add, Mul, Exp, MatMul, etc.).
- Reductions stay (SumDim, MaxDim, ReduceSumTo, ReduceMaxTo, etc.). They're decomposable in principle but have first-class meaning at the IR level — comparable to stablehlo's `reduce`.
- Memory/control stay (Const, Copy, Move, Release).
- View ops stay (Transpose, Permute, BroadcastTo, Slice, Reshape, Unsqueeze).
- Likely additions during/after this refactor: comparison family (Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual), Select/Where, possibly Iota.

### Fused ops move to `FusedOpRegistry`

A struct that owns a `Vec<FusedOpEntry>` (storage) plus several `HashMap` indices for lookup. Indexes keyed by `FusedOpId` (a newtype over `usize`).

```rust
pub struct FusedOpRegistry {
    entries: Vec<FusedOpEntry>,
    by_name: HashMap<&'static str, FusedOpId>,
    by_pattern_hash: HashMap<PatternHash, FusedOpId>,  // for fast fusion matching
    // Backend dispatch is kept on the entries themselves to avoid an
    // (entry, backend) cross-product index when the backend list is small.
}

pub struct FusedOpEntry {
    pub id: FusedOpId,
    pub name: &'static str,
    pub family: FusedOpFamily,           // forward / backward / quantized / attention / norm / ...

    // Identity-by-pattern: the canonical primitive subgraph this fused op
    // represents. Used by fusion rules.
    pub pattern: SubgraphPattern,

    // Decomposition: function that, given the fused-op node's inputs +
    // params, emits a primitive subgraph equivalent. Used by lowering
    // rules and (when backward derivation needs it) by autograd.
    pub decompose: fn(&mut Graph, NodeId, &FusedOpParams) -> NodeId,

    // Backward identity. Either a registered FusedOpId (for fused-backward
    // ops like SoftmaxLastDimBackward) or `Decompose` (autograd derives
    // the backward from the primitive decomposition).
    pub backward: BackwardKind,

    // Per-backend kernel implementations + cost models.
    pub backend_impls: SmallVec<[(BackendId, BackendImpl); 4]>,

    // Shape/dtype rules for graph builders + autograd.
    pub shape_rule: fn(&[Shape], &FusedOpParams) -> Shape,
    pub dtype_rule: fn(&[DType], &FusedOpParams) -> DType,
}

pub struct BackendImpl {
    pub kernel: KernelRef,                  // existing dispatch wrapper signature
    pub cost: fn(&[Shape], &FusedOpParams, &BackendCapabilities) -> CostEstimate,
}

pub struct CostEstimate {
    pub time_ms: f64,                       // estimated wall time
    pub bytes_moved: u64,                   // bandwidth pressure
    pub flops: u64,                         // compute pressure
}

pub enum BackwardKind {
    Fused(FusedOpId),                       // emit this fused op for backward
    Decompose,                              // autograd derives backward from primitive decomposition
    NotDifferentiable,                      // panics in backward (like ArgMaxDim)
}

pub enum FusedOpParams {
    SoftmaxLastDim,
    RmsNormLastDim { eps: f64 },
    LayerNormLastDim { eps: f64 },
    Rope,
    FusedLinear,
    Conv2D { stride: (usize, usize), padding: (usize, usize), groups: usize },
    ConvTranspose2D { /* ... */ },
    FlashAttn { softmax_scale: f32, causal: bool, /* ... */ },
    PagedAttn { /* ... */ },
    QMatMul { quant_type: QuantType, k: usize, n: usize },
    SoftmaxLastDimBackward,
    LayerNormLastDimBackward { eps: f64 },
    RmsNormLastDimBackward { eps: f64 },
    ReduceMaxToBackward,
    // Future: any backend-specific fused op a backend wants to register.
}
```

`FusedOpParams` is itself an enum, but it's the *fused-op-parameter* enum — a different conceptual layer from `Op`. It's not an IR vocabulary; it's the parameter payload for whichever fused op a node represents. Adding a new fused op extends `FusedOpParams` (one variant) plus the registry entry; no other code edits.

### `Node` kind

```rust
pub enum NodeKind {
    Primitive(Op),
    Fused {
        id: FusedOpId,
        params: FusedOpParams,
    },
}

pub struct Node {
    pub kind: NodeKind,
    pub inputs: Vec<NodeId>,
    pub shape: Shape,
    pub dtype: DType,
}
```

The executor's eval_node dispatches:

```rust
match &node.kind {
    NodeKind::Primitive(op) => match op { /* exhaustive primitive dispatch */ },
    NodeKind::Fused { id, params } => {
        let entry = registry.get(*id);
        let backend_impl = entry
            .backend_impls
            .iter()
            .find(|(b, _)| *b == self.backend_id())
            .map(|(_, impl_)| impl_)
            .or_else(|| /* fall back to decomposition */)
        ;
        // dispatch through backend_impl.kernel, params, etc.
    }
}
```

### Cost-based placement query

The Phase 4 scheduler queries the registry per fused-op candidate:

```rust
for candidate in graph.fused_op_nodes() {
    let entry = registry.get(candidate.id);
    for backend in available_backends {
        let cost = if let Some(impl_) = entry.impl_for(backend) {
            impl_.cost(shapes, &candidate.params, &backend.capabilities())
        } else {
            // Decomposed cost: sum primitive op costs for that backend
            cost_of_decomposition(&entry, &candidate.params, backend)
        };
        candidate_costs.insert((candidate, backend), cost);
    }
}
// Run placement optimization over candidate_costs.
```

This is the architectural payoff. The scheduler sees every backend's fusion catalog through one registry interface.

### How this integrates with PR 3's RuleRegistry

PR 3's `RuleRegistry` becomes a thin layer auto-driven by `FusedOpRegistry`:

- For each `FusedOpEntry`, register a `LoweringRule` that emits the entry's `decompose(...)` output.
- For each `FusedOpEntry`, register a `FusionRule` that matches the entry's `pattern` and emits a `Fused { id, params }` node.

The current PR 3 hand-written `SoftmaxLastDimLowerRule` and `SoftmaxLastDimFuseRule` get deleted; `SoftmaxLastDim`'s registry entry generates them. Future fused ops add a registry entry; lowering and fusion rules are free.

Hand-written fusion rules can still exist for patterns that aren't tied to a single FusedOpEntry (e.g., a multi-step canonicalization pass that doesn't end in a fused op). The auto-generated rules are the common case; hand-written are the escape hatch.

---

## Concrete migration path

This is a substantial refactor. Ordered to keep the tree compiling green at every step.

1. **Add `FusedOpRegistry` skeleton** — types, indexes, registration API. No callers yet. Empty registry. fuel-graph compiles.

2. **Add `NodeKind` enum next to `Node.op`** — `Node.op: Op` stays; new `Node.kind: Option<NodeKind>` field. Existing code unaffected.

3. **Pick the first fused op to migrate (SoftmaxLastDim).** Add a registry entry: name, params, pattern, decompose, backward, backend_impls (for CPU initially). Do NOT yet remove `Op::SoftmaxLastDim`. Both representations coexist; the registry entry is unused.

4. **Add the dispatch path for `NodeKind::Fused`.** Executor's eval_node gets a "if node.kind is Some(Fused), dispatch through registry" arm BEFORE the existing `match node.op` arm. Old paths still hit if `kind` is None.

5. **Migrate `Tensor::softmax_last_dim()` builder** to set `node.kind = Some(NodeKind::Fused { id: SOFTMAX_LAST_DIM_ID, params: SoftmaxLastDim })`. Op::SoftmaxLastDim variant stays for backward compatibility but isn't emitted by builders anymore.

6. **Update PR 3 rules.** SoftmaxLastDimLowerRule + SoftmaxLastDimFuseRule become auto-generated from the registry entry. Hand-written versions deleted.

7. **Repeat steps 3-6 for each fused op**: RmsNormLastDim, LayerNormLastDim, Rope, FusedLinear, Conv2D, ConvTranspose2D, FlashAttn, PagedAttn, QMatMul, plus the four backward helpers. ~13 fused ops.

8. **Drop `Op` variants**. Once nothing emits `Op::SoftmaxLastDim` etc., remove the variants from `Op`. Update op_short_name, op_key, autograd's match-on-Op (which now panics on the removed variants — but they're unreachable).

9. **Remove `Node.op`, rename `Node.kind` to canonical**. The old `Op` field is dead; remove it. NodeKind becomes the only op-identity field.

10. **Backend registrations migrate**. fuel-storage's `register_*_kernels` functions move to "for each FusedOpEntry, look up backend_impl, register against the binding table." Same kernel wrappers; different registration surface.

11. **Cost models populated**. Each `FusedOpEntry`'s `cost` function gets a real implementation per backend (initially: simple FLOP-counting + bandwidth model; refined as the scheduler arrives).

Each step is a separable PR. Steps 1-2 are mechanical. Steps 3-7 are the real work — ~13 fused ops × ~half-day each, plus ~2 days of executor and binding-table refactor. Steps 8-9 are mechanical cleanup once nothing uses the old variants. Steps 10-11 land alongside the Phase 4 scheduler.

Estimated total: 2-3 weeks for the full migration; partial migration (steps 1-7 for one fused op) is shippable as a proof of concept in 3-4 days.

---

## Open design questions

1. **Where does `FusedOpRegistry` live?** Options: `fuel-graph` (alongside Op), `fuel-core-types` (alongside DType / dispatch types), or a new `fuel-fused-ops` crate. Leaning fuel-graph for cohesion with Op, but a new crate is cleaner if the registry needs to depend on backend types for kernel signatures.

2. **`FusedOpParams` as an enum vs `Box<dyn Any>`.** Enum gives compile-time exhaustiveness but requires editing the enum for every new fused op (the same friction we're trying to escape for `Op`). `Box<dyn Any>` is fully dynamic but loses type safety. Recommendation: enum for now (we're not adding fused ops at runtime; registration is at startup). Re-evaluate if a downstream consumer needs runtime-extensible fused ops.

3. **Pattern representation.** PR 3's hand-written matchers walk back from a pivot node. The registry needs the same machinery in declarative form. Options: a small pattern DSL (a la egg's RecExpr), an `Fn(&Graph, NodeId) -> Option<Match>` closure per entry, or a structural pattern struct (op-and-children). Recommendation: closure + helper functions for common shapes; if pattern proliferation grows past ~30 entries, revisit the DSL question.

4. **Cost estimate stability.** Per-backend cost functions need shape + dtype + backend capabilities; how stable is the `BackendCapabilities` interface across the migration? Likely needs to grow as the cost models mature — start with simple "FLOPs / peak-FLOPs + bytes / peak-bandwidth" and extend as needed.

5. **How does `OpKind` evolve?** Currently `OpKind` is a closed enum mirroring `Op` for binding-table dispatch. Post-migration, the binding table is keyed by either `OpKind` (for primitives) or `FusedOpId` (for fused ops). Either two parallel binding tables, or one unified key. Recommendation: unified key `DispatchKey { Primitive(OpKind), Fused(FusedOpId) }`; mechanical change.

6. **Backend-specific fused ops that don't have a cross-backend abstraction.** Example: a CUDA-only `MatMul + Sigmoid` fusion that no other backend cares about. Does it get a FusedOpId visible to all backends (just a CUDA `BackendImpl` and no others)? Yes — that's how the registry naturally encodes "backend X has a fused kernel for this pattern, others fall back to decomposition." The cost model for non-implementing backends uses the decomposition cost. Open question: if 50 such backend-specific fusions land, does the registry accommodate them gracefully or do we need a "scope" concept (private vs shared FusedOpIds)? Defer until we have a real consumer.

7. **Migration order for backward helpers.** Should `SoftmaxLastDimBackward` etc. migrate alongside their forward op (paired registration) or separately? Recommendation: paired. Each fused-op entry has a `backward: BackwardKind` field; if it's `BackwardKind::Fused(id)`, the backward op gets its own entry registered alongside the forward.

---

## Out of scope (this work item)

- **Cost-based scheduler implementation.** The scheduler that consumes per-backend cost estimates is a separate phase (Phase 4 scheduler refinement). This refactor produces the *substrate* the scheduler will query.
- **Multi-level dialect IR.** `Primitive(Op) | Fused(FusedOpId)` is two layers, not full MLIR-style multi-dialect. Adequate for fuel's needs.
- **Pattern-match autotuning / e-graph equality saturation.** Anchored structural matching (PR 3's style) is sufficient. e-graphs are a future option if pattern complexity grows.
- **User-extensible fused ops at runtime.** Registry is populated at startup, frozen thereafter. Hot-add isn't a goal.
- **Bool dtype.** Comparison-op output is float (1.0/0.0) per the comparison-family decision. Bool dtype is independent and orthogonal.

---

## Honest caveats

- **This refactor touches the deepest layer of fuel.** Every executor, every backend, every autograd path matches on `Op`. Those matches all change shape. Mitigation: the parallel-introduction-then-drop tactic — `Node.op: Op` and `Node.kind: NodeKind` coexist throughout the migration window, with a drop step at the end. Each individual fused-op migration is independently shippable.

- **Auto-generated rules vs hand-written rules.** PR 3 shipped hand-written rules that are easy to read and reason about. Auto-generated rules from registry entries are more abstract — debugging a misbehaving fusion requires understanding the rule generator, not just the rule. Mitigation: keep PR 3's hand-written form available as a "manual override" escape hatch when auto-generation isn't sufficient.

- **Cost estimates can mislead the scheduler.** A FLOP-counting cost model gets the rough magnitudes right but misses fixed launch overhead, kernel-launch latency on busy queues, memory-bandwidth saturation interactions, etc. Mitigation: cost estimates are advisory; the scheduler also measures actual runtimes and adapts (Phase 6b probe/judge/dispatch pattern). Initial cost models can be coarse; they get refined as profile data accumulates.

- **The Op enum gets harder to extend mid-migration.** Adding a new primitive Op variant during the migration is fine; adding a new "high-level" op forces a choice (variant or registry entry?). Mitigation: during the migration window, default new high-level ops to registry entries; new primitives to `Op` variants. The migration's first step (the registry skeleton) makes this choice possible from day one.

- **Backend authors need to learn a new registration surface.** `register_softmax_kernel(backend, dtype, fn)` becomes "look up the SoftmaxLastDim FusedOpEntry, attach a BackendImpl." This is genuinely better but new. Mitigation: a registration macro that hides the boilerplate; backend authors mostly write `register_fused!(softmax_last_dim, cuda, f32, my_kernel)`.

---

## Success criteria

- `Op` enum is primitive-only (~85 variants, mostly stable). No fused-op variants remain.
- `FusedOpRegistry` is populated with ~13 entries (the existing fused ops). Adding a new fused op is a registry entry + a kernel function, no edits to `Op` / autograd / executor.
- PR 3's `SoftmaxLastDimLowerRule` + `SoftmaxLastDimFuseRule` are deleted. Auto-generated rules from the registry produce equivalent behavior. Round-trip identity test still passes.
- Live CUDA equivalence test (`cuda_executor_matches_cpu_on_softmax_via_lowering`) still passes — the lowered subgraph runs natively on CUDA.
- A `cost_estimate(SoftmaxLastDim, [B,N,M], CUDA)` query returns a plausible time estimate (not the scheduler proper; the registry surface for it).
- All existing tests green throughout the migration. fuel-graph CSE / op_key handles `NodeKind::Fused` correctly (e.g., two fused nodes with the same id and params CSE to one).
- `MEMORY.md` and `ROADMAP.md` updated with the post-migration architecture summary.

---

## References

- **Stablehlo op set**: https://github.com/openxla/stablehlo (~80 primitive ops; reference for what fuel's primitive layer should contain).
- **MLIR dialects**: https://mlir.llvm.org/docs/Dialects/ (industry analog for the open-registry-of-closed-enums pattern).
- **PR 3 rule registry**: `fuel-graph/src/opt.rs` (`Rule`, `RuleFamily`, `RuleRegistry`) — substrate this refactor builds on.
- **Storage unification**: `docs/storage-unification.md` — sibling refactor with a similar parallel-introduction-then-drop migration tactic.
