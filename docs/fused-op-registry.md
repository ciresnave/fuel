# Phase 7.6 — FusedOpRegistry implementation design (v3)

**Status**: design v3, 2026-05-09. Anchored to architecture v1.0 (see [`docs/architecture/`](architecture/00-index.md)). Steps 1-3 shipped on `feature/storage-unification` (commits `408ff57a`, `e15f0ce9`, `10f04b87`); step 4+ pending.

This document is the implementation-side design for Phase 7.6. Architectural commitments live in the architecture set; this document carries the *how* — type shapes, file layouts, migration steps, code-shape examples, open implementation questions.

**v3 corrects v2's crate-placement of `FusedOpEntry::backend_impls`**: v2 wrote a single `FusedOpEntry` struct in `fuel-graph` carrying `SmallVec<[(BackendId, BackendImpl); 4]>`, but `BackendImpl` carries `KernelRef` (in `fuel-storage`) and `fuel-storage` already depends on `fuel-graph`, not the reverse. The correct shape — described in architecture v1.0 §03-ir's "What lives where" table but only loosely in v2 — is a **two-half registry joined by `FusedOpId`**: metadata in `fuel-graph::registry`, kernel payloads in `fuel-storage::fused`. v3 makes the split explicit and updates the type shapes + migration steps accordingly. Surfaced during step-1 implementation 2026-05-09.

**v2 superseded the original Phase 7.6 design** (which used a `NodeKind::{Primitive | Fused}` discriminator and treated the registry crate location as an open question). Architecture v1.0 made several decisions that change the implementation-side picture:

- **Op-shape A locked**: single `Op` enum with primitive variants + one `Op::Fused(FusedOpId, FusedOpParams)` arm. No separate `NodeKind` discriminator type.
- **Pre-resolved `KernelRef` per node**: binding table is a planning-time catalog, not a runtime lookup.
- **Per-decision-point alternatives**: optimizer output is alternative sets per decision point, not N global routes.
- **`PrecisionGuarantee` per kernel**: replaces the OracleGrade flag; structured per-kernel precision metadata.
- **Cache + telemetry infrastructure**: persistence layer + community telemetry are first-class; the registry feeds both.

The v1 design's ROADMAP entry has been rewritten to match v2; this document is the corresponding implementation guide.

---

## TL;DR

Today's `Op` enum is a hybrid: ~60 variants mixing primitives (Add, Mul, Exp, MatMul) with fused abstractions (SoftmaxLastDim, RmsNormLastDim, FlashAttn). Adding a new fused op multiplies plumbing across every backend, every autograd path, every dispatch wrapper.

**The split**: `Op` becomes primitive variants + one `Op::Fused(id, params)` arm. The arm indexes a build-time-frozen, runtime-immutable `FusedOpRegistry` of fused-op entries. Each entry encodes its primitive subgraph signature (for fusion pattern recognition), its decomposition (for lowering), per-backend kernel implementations with cost estimates and `PrecisionGuarantee` metadata (for placement and tolerance reasoning), and its backward op (anchored as another `FusedOpId` or as a primitive subgraph).

Adding a new fused op: one registry entry + one kernel function per backend that supports it. No `Op` enum edit beyond the existing `Fused` arm; no autograd edit; no per-backend executor arm; no `op_short_name`/`op_key` edit.

---

## Goals

- **Closed primitive set, open fused-op registry.** `Op`'s primitive variants stay small, exhaustively matched, panic-free. Fused ops are added without touching the primitive variants.
- **Cross-backend fusion visibility.** Every backend's fused-kernel catalog is visible to the optimizer for cost-based placement.
- **Bidirectional pattern↔fused-op mapping.** Lowering: `FusedOpId` → primitive subgraph. Fusion: primitive subgraph pattern → `FusedOpId`. Same registry, two indices.
- **One source of truth per fused op.** Each entry defines its decomposition, pattern, backend impls, and backward — no risk of drift between Op-variant docs and lowering-rule definitions.
- **Backend extensibility.** A backend adds a fused kernel by registering a `BackendImpl` against a `FusedOpId`. No `Op` enum edit, no autograd edit, no executor arm edit.
- **Pre-resolved KernelRef per node.** The binding table is consulted at planning time (per-decision-point pick + lazy resolution); the executor calls function pointers directly.
- **Per-kernel `PrecisionGuarantee`.** Each `BackendImpl` declares its precision properties; the optimizer uses them for tolerance-budget admissibility and for selecting calibration comparators.

## Non-goals (this work item)

- **Cost-based scheduler implementation.** This refactor enables it; the actual scheduler that consumes per-backend cost estimates is downstream phase work.
- **e-graph / equality-saturation pattern matching.** PR 3's anchored structural matching + declarative-pattern engine (per architecture's OptimizationMap rule shape) is sufficient for v1.
- **Multi-level dialect IR (MLIR-style).** Two layers — primitive `Op` variants + fused-op registry behind `Op::Fused` — covers fuel's needs.
- **Backend-specific autotuning.** Cost estimates come from per-backend `BackendImpl.cost` functions; how a backend computes them (static, profile-driven, autotuner) is a backend concern.
- **Runtime-extensible registry.** The registry is populated at process startup, frozen thereafter. No hot-add.

---

## Type shapes

### `Op` enum (post-migration)

In `fuel-graph`:

```rust
pub enum Op {
    // ~80 primitive variants, closed and exhaustive.
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Conv2D { stride: (usize, usize), padding: (usize, usize), groups: usize },
    BroadcastTo(Shape),
    Permute(Vec<usize>),
    Slice { dim: usize, start: usize, end: usize, step: usize },
    Cast(DType),
    // ... etc.

    // One arm for fused ops. Adds a new fused op via the registry, not via the enum.
    Fused(FusedOpId, FusedOpParams),
}
```

### Registry types — split across two crates, joined by `FusedOpId`

The registry is two halves: graph-side metadata in `fuel-graph::registry` and kernel-side payload in `fuel-storage::fused`. The split exists because `KernelRef` lives in `fuel-storage` (which already depends on `fuel-graph`), so a single struct holding both pattern callables and `KernelRef` cannot live in either crate without inverting the dependency. `FusedOpId` is the runtime join key: the optimizer reads the metadata-side entry to reason about decomposition / shape / backward, then asks the kernel-side `FusedKernelRegistry` for the per-backend `BackendImpl` when it needs to pre-resolve a `KernelRef`.

#### Graph-side metadata in `fuel-graph::registry`

```rust
pub struct FusedOpId(pub u16);  // newtype; ~65K capacity is plenty

impl FusedOpId {
    pub const UNASSIGNED: FusedOpId = FusedOpId(0);  // reserved sentinel; slot 0
}

pub struct FusedOpRegistry {
    entries:         Vec<FusedOpEntry>,                  // id-indexed; slot 0 = UNASSIGNED placeholder
    by_name:         HashMap<&'static str, FusedOpId>,
    by_pattern_hash: HashMap<PatternHash, FusedOpId>,    // reserved for the step-4 declarative pattern engine
}

pub struct FusedOpEntry {
    pub id:     FusedOpId,
    pub name:   &'static str,
    pub family: FusedOpFamily,           // forward / backward / quantized / attention / norm / ...

    /// Identity-by-pattern: the canonical primitive subgraph this fused op
    /// represents. Used by fusion rules.
    pub pattern: SubgraphPattern,

    /// Decomposition: function that, given the fused-op node's inputs +
    /// params, emits a primitive subgraph equivalent. Used by lowering
    /// rules and (when backward derivation needs it) by autograd.
    pub decompose: fn(&mut Graph, NodeId, &FusedOpParams) -> NodeId,

    /// Backward identity. Either a registered `FusedOpId` (for fused-backward
    /// ops like SoftmaxLastDimBackward), `Decompose` (autograd derives
    /// the backward from the primitive decomposition), or `NotDifferentiable`.
    pub backward: BackwardKind,

    /// Shape/dtype rules for graph builders + autograd + cost evaluation.
    pub shape_rule: fn(&[Shape], &FusedOpParams) -> Shape,
    pub dtype_rule: fn(&[DType], &FusedOpParams) -> DType,

    // Per-backend kernel implementations live in fuel-storage::fused
    // (FusedKernelRegistry, keyed by FusedOpId). They are NOT a field on
    // this struct because BackendImpl carries KernelRef which lives in
    // fuel-storage (which already depends on fuel-graph). See "split"
    // note above.
}

pub enum FusedOpParams {
    SoftmaxLastDim,                      // step 3 (shipped)
    RmsNormLastDim       { eps: f64 },   // step 4
    LayerNormLastDim     { eps: f64 },   // step 4
    Rope,                                // step 4
    FusedLinear,                         // step 4
    Conv2D               { stride: (usize, usize), padding: (usize, usize), groups: usize },  // step 4
    ConvTranspose2D      { /* ... */ },  // step 4
    FlashAttn            { softmax_scale: f32, causal: bool, /* ... */ },  // step 4
    PagedAttn            { /* ... */ },  // step 4
    QMatMul              { quant_type: QuantType, k: usize, n: usize },    // step 4
    SoftmaxLastDimBackward,                                                 // step 4
    LayerNormLastDimBackward { eps: f64 },                                  // step 4
    RmsNormLastDimBackward   { eps: f64 },                                  // step 4
    ReduceMaxToBackward,                                                    // step 4
    // Future fused ops add a variant here. The variant is the single point
    // of growth in fuel-graph for fused-op extension.
}

pub enum BackwardKind {
    Fused(FusedOpId),         // emit this fused op for backward
    Decompose,                 // autograd derives backward from primitive decomposition
    NotDifferentiable,         // panics in backward (like ArgMaxDim)
}

pub enum SubgraphPattern {
    Declarative(PatternTree),                                       // step 4 wires the engine
    Callable(fn(&Graph, NodeId) -> Option<PatternMatch>),           // step 3 ships this arm
}

pub struct PatternMatch {
    pub bindings: Vec<(usize, NodeId)>,   // var-id → resolved NodeId
}

/// Process-wide default registry. Built once via `OnceLock`; immutable thereafter.
pub fn default_registry() -> &'static FusedOpRegistry { /* ... */ }
```

#### Kernel-side payload in `fuel-storage::fused`

```rust
pub struct BackendImpl {
    pub kernel:    KernelRef,                                                    // existing dispatch wrapper signature
    pub cost:      fn(&[Shape], &FusedOpParams, &BackendCapabilities) -> CostEstimate,
    pub precision: PrecisionGuarantee,                                            // per architecture v1.0
    pub caps:      KernelCaps,                                                    // existing capability flags
    pub revision:  KernelRevisionHash,                                            // for cache invalidation
}

pub struct CostEstimate {
    pub flops:               u64,    // compute pressure
    pub bytes_moved:         u64,    // bandwidth pressure
    pub kernel_overhead_ns:  u32,    // launch latency
}

pub struct PrecisionGuarantee {  // see docs/architecture/05-backend-contract.md
    pub bit_stable_on_same_hardware: bool,
    pub max_ulp:      Option<u32>,
    pub max_relative: Option<f64>,
    pub max_absolute: Option<f64>,
    pub notes:        &'static str,
}

impl PrecisionGuarantee {
    pub const REFERENCE: Self = /* ... */;   // bit-stable, ULP=0; reference IEEE-754
    pub const UNKNOWN:   Self = /* ... */;   // step-7 lint replaces every UNKNOWN with a real claim
}

pub struct KernelRevisionHash(pub u64);

impl KernelRevisionHash {
    pub const UNTRACKED: Self = KernelRevisionHash(0);  // step-9 wires real hashing
}

/// Kernel-side registry: FusedOpId → list of per-backend BackendImpls.
/// Joined to fuel-graph::registry::FusedOpRegistry by id at runtime.
pub struct FusedKernelRegistry {
    by_id: HashMap<FusedOpId, SmallVec<[(BackendId, BackendImpl); 4]>>,
}

impl FusedKernelRegistry {
    pub fn register(&mut self, id: FusedOpId, backend: BackendId, impl_: BackendImpl);
    pub fn lookup(&self, id: FusedOpId, backend: BackendId) -> Option<BackendImpl>;
    pub fn impls_for(&self, id: FusedOpId) -> &[(BackendId, BackendImpl)];
}
```

### `Node`

Unchanged structure (no `NodeKind` wrapper):

```rust
pub struct Node {
    pub op:     Op,           // primitive variant OR Op::Fused(id, params)
    pub inputs: Vec<NodeId>,
    pub shape:  Shape,
    pub dtype:  DType,
}
```

### Optimizer's per-decision-point alternative set

Per architecture v1.0, the optimizer's output is per-decision-point alternatives. Each decision point's alternative carries a pre-resolved `KernelRef` (lazy at first use) plus metadata:

```rust
pub struct DecisionPointAlternative {
    pub plan_subgraph: NodeId,                    // root of the alternative's subgraph
    pub kernel_choices: Vec<NodeKernelBinding>,    // pre-resolved per node in the alternative
    pub cost_estimate: CostEstimate,               // composed cost
    pub cumulative_error: ErrorEstimate,           // for tolerance budget tracking
    pub frontier_compat: FrontierCompatibility,    // Concurrent | WholeGraph
}

pub struct NodeKernelBinding {
    pub node:               NodeId,
    pub kernel:              Option<KernelRef>,    // lazy: None until first use; resolved via binding-table lookup
    pub backend:             BackendId,
    pub device:              DeviceLocation,
    pub kernel_revision:     KernelRevisionHash,    // recorded for cache invalidation
}
```

Implementation detail; included here so backend authors and rule authors see the shape their `BackendImpl`s end up populating.

---

## How rules use the registry

PR 3's hand-written rules become auto-generated from registry entries:

- For each `FusedOpEntry`, register a **lowering rule** that emits the entry's `decompose(...)` output.
- For each `FusedOpEntry`, register a **fusion rule** that matches the entry's `pattern` and emits an `Op::Fused(id, params)` node.

Concrete generator:

```rust
impl FusedOpEntry {
    pub fn lowering_rule(&self) -> Box<dyn Rule> {
        Box::new(LoweringRule {
            id:         self.id,
            decompose:  self.decompose,
            params_for: ...,  // extracts FusedOpParams from a matched Op::Fused node
        })
    }

    pub fn fusion_rule(&self) -> Box<dyn Rule> {
        Box::new(FusionRule {
            id:      self.id,
            pattern: self.pattern.clone(),
        })
    }
}
```

PR 3's hand-written `SoftmaxLastDimLowerRule` and `SoftmaxLastDimFuseRule` get deleted; the registry entry produces equivalent behavior. Hand-written rules remain available as an escape hatch for canonicalization passes that don't terminate in a single fused op.

---

## How the executor consumes the registry

Per architecture v1.0, the executor calls pre-resolved `KernelRef` function pointers directly. It never looks up kernels at execution time. The registry is consulted by the *optimizer* (when populating decision-point alternatives) and by *autograd* (when emitting backward).

Executor's per-node dispatch (post-migration):

```rust
fn execute_node(node: &Node, alt: &NodeKernelBinding, inputs: &[Storage], outputs: &mut [Storage]) {
    let kernel = alt.kernel.expect("KernelRef pre-resolved by route picker");
    kernel(inputs, outputs, &node.layouts, &node.params).expect("kernel returned Result::Err")
}
```

Compare to today's binding-table lookup at execution time:

```rust
// PRE-MIGRATION (today):
let kernel = binding_table.lookup(node.op_kind(), node.dtypes(), backend_id)?;
kernel(inputs, outputs, &node.layouts, &node.params)?;
```

Today's lookup happens per node per realize. Post-migration, the route picker resolves once per decision point per realize (lazy: only when an alternative is picked); the executor never looks up.

---

## Migration path

Eleven steps. Each is independently shippable. Tree compiles green at every commit boundary.

### Step 1: registry skeleton (no callers) — **shipped 2026-05-09 (`408ff57a`)**

In `fuel-graph::registry` (graph-side metadata):

- `FusedOpId(u16)` newtype + `FusedOpId::UNASSIGNED` sentinel.
- `FusedOps` associated-constants struct (`SOFTMAX_LAST_DIM = FusedOpId(1)`).
- `FusedOpRegistry` struct with id-indexed `entries` Vec + `by_name` index + reserved `by_pattern_hash` index.
- `FusedOpEntry` struct (without `backend_impls` — see split note above).
- `FusedOpParams` enum (start with one variant: `SoftmaxLastDim`; extend per migration).
- `FusedOpFamily`, `BackwardKind`, `SubgraphPattern { Declarative(PatternTree), Callable(fn) }`, `PatternMatch`, `PatternTree` (placeholder), `FusedOpParamsKey` (for CSE/op_key dedup).

In `fuel-storage::fused` (kernel-side payload):

- `BackendImpl` struct.
- `PrecisionGuarantee` struct (per architecture v1.0) with `REFERENCE` and `UNKNOWN` consts.
- `CostEstimate` struct.
- `KernelRevisionHash` newtype with `UNTRACKED` sentinel.
- `FusedKernelRegistry` struct (id → `SmallVec<[(BackendId, BackendImpl); 4]>`).

No callers; types compile; no behavior change. Tree green.

### Step 2: extend `Op` enum with `Op::Fused(FusedOpId, FusedOpParams)` arm — **shipped 2026-05-09 (`e15f0ce9`)**

Added the variant to `Op`. Existing variants (`Op::SoftmaxLastDim`, etc.) coexist with the new arm during migration. Updated `op_short_name`, `op_key` (CSE keyed on `(id, FusedOpParamsKey)`), autograd's match (panic stub until step 3), and the four exhaustive executor consumers (`fuel-graph-executor`, `fuel-graph-cpu`, `fuel-reference-backend` — `unreachable!()` arms; `fuel-storage::pipelined::op_to_op_kind` and `op_to_op_params` — wildcard catch-all needed no edit).

Tree compiles green; no behavior change yet (no nodes use `Op::Fused`).

### Step 3: migrate first fused op (SoftmaxLastDim) end-to-end — **shipped 2026-05-09 (`10f04b87`)**

The proof-of-concept commit. After this step, one fused op flows through the registry; the others use the legacy variants.

- Created the SoftmaxLastDim registry entry in `fuel-graph::registry::softmax_last_dim`: pattern (`SubgraphPattern::Callable`), decompose fn (port of PR 3's `SoftmaxLastDimLowerRule::rewrite`), shape/dtype passthrough rules, `BackwardKind::NotDifferentiable` (the *backward* fused-op itself migrates in step 4; `Tensor::backward` dispatches the SoftmaxLastDim arm directly until then).
- Process-wide `default_registry()` factory (`OnceLock`-backed).
- Auto-generated `LoweringRule` and `FusionRule` types in `fuel-graph::opt` that read `decompose` / pattern from a `FusedOpEntry`. `RuleRegistry::default_rules` and `RuleRegistry::lowering_only` iterate `default_registry().entries_iter()` and produce one rule pair per registered fused op.
- Deleted PR 3's hand-written `SoftmaxLastDimLowerRule` and `SoftmaxLastDimFuseRule`.
- `Tensor::softmax_last_dim()` builder emits `Op::Fused(FusedOps::SOFTMAX_LAST_DIM, FusedOpParams::SoftmaxLastDim)` instead of `Op::SoftmaxLastDim`. Legacy variant stays in the enum during migration; step 5 drops it.
- `Tensor::backward` per-id arm: `Op::Fused(SOFTMAX_LAST_DIM, _)` emits `Op::SoftmaxLastDimBackward` (legacy variant); the proper `BackwardKind::Fused(SOFTMAX_LAST_DIM_BACKWARD)` connection lands when the backward fused-op migrates in step 4.
- Executor dispatch arms in `fuel-graph-executor`, `fuel-graph-cpu`, `fuel-reference-backend`: each routes `Op::Fused(SOFTMAX_LAST_DIM, _)` to the same softmax-last-dim kernel as the legacy variant. Step-3 bridge pattern; step 9 replaces these with pre-resolved KernelRefs from the route picker.
- `fuel-storage::pipelined::op_to_op_kind` + `op_to_op_params`: both shapes resolve to `OpKind::SoftmaxLastDim` and `OpParams::SoftmaxLastDim`, so existing per-dtype CPU/CUDA wrappers continue to handle dispatch unchanged.

Tree compiles green; live CUDA equivalence test (`cuda_executor_matches_cpu_on_softmax_via_lowering`) passes via the registry-dispatched path (max abs err `4.47e-8` vs `1e-5` tolerance). **This is the natural pause point if the session needs to end early.**

#### Honest caveats from step 3 (carry into step 4)

- `FusionRule::rewrite` reconstructs `FusedOpParams` by hard-coding the SoftmaxLastDim variant. Step 4 generalizes — either by extending `PatternMatch` with a params-binding field, or by adding a sibling `extract_params: fn(&Graph, &PatternMatch) -> FusedOpParams` to `FusedOpEntry`.
- `LoweringRule` continues to fire on the legacy `Op::SoftmaxLastDim` variant (alongside `Op::Fused(SOFTMAX_LAST_DIM, _)`) so emission sites that haven't migrated (the pipelined-executor test that constructs the node directly) keep working. The legacy fallback comes out with step 5.
- `BackwardKind::Fused(...)` is wired but unused: the SoftmaxLastDim entry uses `BackwardKind::NotDifferentiable` because `Tensor::backward` dispatches the registry arm directly. The proper `BackwardKind::Fused(SOFTMAX_LAST_DIM_BACKWARD)` connection lands when the backward fused-op migrates in step 4.

### Step 4: migrate remaining 12 fused ops

Each is its own commit. Repeat the step-3 pattern for: RmsNormLastDim, LayerNormLastDim, Rope, FusedLinear, Conv2D, ConvTranspose2D, FlashAttn, PagedAttn, QMatMul, plus the 4 backward-helper fused ops.

For each: registry entry, executor dispatch arm, builder migration, auto-generated rules, delete hand-written rules (where any).

~half-day per op; ~6 days total.

### Step 5: drop per-fused-op `Op` variants

Once nothing emits `Op::SoftmaxLastDim`, `Op::RmsNormLastDim`, etc., remove them from the enum. Mechanical:

- Remove variants from `Op`.
- Remove arms from `op_short_name`.
- Remove arms from `op_key`.
- Remove arms from autograd's match (the variants are unreachable anyway; rustc requires they be removed once dropped from `Op`).

Tree compiles green; no behavior change (no node was reaching the dropped arms).

### Step 6: backend registrations adopt `BackendImpl` shape

fuel-storage's `register_*_kernels` functions (currently `register(table, OpKind, dtypes, backend, kernel)`) extend to "for each FusedOpEntry, attach a BackendImpl containing kernel + cost + PrecisionGuarantee." Macro hides boilerplate:

```rust
register_fused!(
    softmax_last_dim,
    cuda,
    &[F32],
    cuda_softmax_last_dim_f32,
    cost = cost_softmax_cuda,
    precision = PrecisionGuarantee {
        bit_stable_on_same_hardware: false,
        max_ulp: Some(2),
        max_relative: Some(1e-6),
        max_absolute: None,
        notes: "Uses CUDA's __expf intrinsic; bounded ULP error.",
    },
);
```

For primitive ops, the existing binding-table-style registration continues to work; the macro is the new path for fused-op registrations specifically.

### Step 7: populate `PrecisionGuarantee` per registered kernel

For every kernel registered in steps 4-6, declare the `PrecisionGuarantee`. The always-built backend (fuel-cpu-backend by convention) commits to providing at least one `bit_stable_on_same_hardware: true` kernel per primitive op as the architecture v1.0 coverage commitment. Add a CI lint asserting this coverage.

**Status (2026-05-11, commit f2e5a45f via fuel-storage)**: Step 7a (the **fused-op half**) is **shipped** — `precision_guarantee_lint_bit_stable_cpu_coverage` in `fuel-storage::fused::tests` iterates every entry in `fuel-graph::registry::default_registry()` and asserts at least one `bit_stable_on_same_hardware: true` CPU `BackendImpl` exists in `fuel-storage::fused::default_kernel_registry()`. Today 10 of 14 entries are covered; the 4 backward helpers (SoftmaxLastDimBackward / LayerNormLastDimBackward / RmsNormLastDimBackward / ReduceMaxToBackward) appear on a `KNOWN_GAPS` allowlist with documented reasons — their CPU dispatch flows through `GraphBackend` trait methods rather than byte-level binding-table wrappers, so step-6 registration awaits either wrapper conversion or step-9's trait-method-as-KernelRef path.

**Step 7b — primitive-op coverage extension — is pending.** The lint covers the *fused-op registry only*. Primitive ops (`Op::Add`, `Op::MatMul`, view ops, etc.) dispatch through `KernelBindingTable` in `fuel-storage::dispatch`, which doesn't carry a `PrecisionGuarantee` field today. Extending the architecture commitment to primitives needs either a binding-table refactor (add `PrecisionGuarantee` per registration site) or a parallel `PrecisionGuarantee` side-table keyed by `(OpKind, dtype-tuple, BackendId)`. The lint's docstring (`fuel-storage::fused::tests::precision_guarantee_lint_bit_stable_cpu_coverage`) calls this out so it stays visible.

### Step 8: populate cost estimates

Each `BackendImpl`'s `cost` function gets a real implementation. Initial: FLOP-counting + bandwidth model (the conservative static-only form). The community-aggregated empirical refinement framework (per [11-persistence §Cache generation and distribution](architecture/11-persistence.md#cache-generation-and-distribution)) tightens these over time as telemetry pipeline lands.

### Step 9: binding-table planning-time refactor

Migrate per-kernel binding-table lookup off the executor's hot path:

- The route picker pre-resolves `KernelRef` at decision-point pick time (lazy: only when an alternative is selected).
- The executor calls the pre-resolved function pointer directly; never looks up.
- The binding table becomes a planning-time catalog only.

This resolves audit Q-A and is the foundation for [11-persistence §Re-resolution on use](architecture/11-persistence.md#re-resolution-on-use-lazy-not-at-load) (lazy resolution + mmap'd cache). The cache work itself is downstream phase work.

### Step 10: comparison family added as primitive variants

Add Equal/NotEqual/Less/LessEqual/Greater/GreaterEqual to `Op` as primitive variants. Bit-exact equality on floats; non-differentiable backward (panic stub, ArgMaxDim precedent). Lands in this phase because primitive-set completion belongs with this architectural cleanup.

### Step 11: update memory + ROADMAP

- Update `MEMORY.md` to reflect post-migration architecture.
- Update ROADMAP Phase 7.6 entry to mark complete.
- Add a decisions-log entry to `docs/architecture/10-decisions-log.md` if any architectural commitment changed (none expected; but procedural).

---

## Cross-cutting concerns

### Layout side-table stays single source of truth

PR 3's Layout-on-Node migration completed before this phase. `Op::is_view_op()` still answers based on the variant; the `Op::Fused` arm is never a view op (none of the 13-14 registered fused ops are layout-only operations). `Graph::push`'s auto-populate logic for view ops is unchanged.

### CSE / op_key handles `Op::Fused`

Two `Op::Fused(id, params)` nodes with the same id and the same params should CSE to one node (same architectural property as today's variant CSE). Implementation: `op_key` for the `Fused` arm returns a key derived from `(id, hash(params))`. Standard.

### Autograd backward dispatch

Today autograd has an inline match-on-Op (`Tensor::backward`'s ~600-line match). Post-migration:

- For primitive variants: same as today; the match arm dispatches per-primitive-rule.
- For `Op::Fused(id, _)`: look up `registry.entry(id).backward`; dispatch per `BackwardKind`:
  - `Fused(backward_id)`: emit an `Op::Fused(backward_id, ...)` node.
  - `Decompose`: invoke `entry.decompose(...)` to expand the primitives, then run autograd over the primitive subgraph (graph-rewrite-as-backward).
  - `NotDifferentiable`: panic with a clear message (matches today's QMatMul / ArgMaxDim treatment).

The 3 already-migrated `GradientRule` impls (Add, Mul, Relu — primitives) are unaffected. The 4 fused-backward-helper ops (SoftmaxLastDimBackward, LayerNormLastDimBackward, RmsNormLastDimBackward, ReduceMaxToBackward) become registry entries with `BackwardKind::NotDifferentiable` (matching today's higher-order-gradient panic).

### Per-decision-point alternatives integration

The optimizer's output is per-decision-point alternative sets. The registry feeds this:

- For each fused-op node in the graph, the optimizer can keep the fused form OR keep the decomposition (via the registry's `decompose` function) as alternatives at the decision point.
- For each subgraph that matches a registered pattern, the optimizer can fuse to the registered op OR leave decomposed as alternatives.
- The route picker resolves at dispatch time per [04-optimization §Per-decision-point alternatives](architecture/04-optimization.md#per-decision-point-alternatives).

### `KernelCaps` continues to apply per-kernel

The `BackendImpl.caps` field carries the existing `KernelCaps` (per [03-ir §Layout](architecture/03-ir.md#layout-a-side-table-not-metadata-on-storage)) — `strided_input` and future capability flags. The optimizer's layout-fixup pass reads these to decide whether `Op::Contiguize` insertions are needed.

---

## Open implementation questions

These are bounded; the architecture has resolved the bigger design choices.

### Q1: How is `SubgraphPattern` represented?

Two reasonable shapes:

- **Closure-based**: `SubgraphPattern = fn(&Graph, NodeId) -> Option<Match>`. PR 3's matchers are this shape. Maximally flexible; less analyzable.
- **Declarative tree-pattern with variables**: a recursive struct `Pattern::Op(Op, Vec<Pattern>) | Pattern::Var(VarId)` that the rule engine compiles to a matcher. More analyzable; auto-generation of the matcher from the registry entry's pattern is straightforward; matches architecture's "declarative + callable engine" commitment from [04-optimization](architecture/04-optimization.md#optimizationmap).

**Recommendation**: support both. Most fused-op patterns are simple enough for the declarative form; PR 3's `SoftmaxLastDimFuseRule`-style consumer-count guards stay in callable closures. The `SubgraphPattern` enum carries either:

```rust
pub enum SubgraphPattern {
    Declarative(PatternTree),
    Callable(fn(&Graph, NodeId) -> Option<Match>),
}
```

### Q2: How are `FusedOpId` constants assigned and accessed?

Constants are assigned at registry initialization time. To make pattern-matching ergonomic in rule code, expose them as associated constants on a `FusedOps` struct:

```rust
impl FusedOps {
    pub const SOFTMAX_LAST_DIM:    FusedOpId = FusedOpId(1);
    pub const RMS_NORM_LAST_DIM:   FusedOpId = FusedOpId(2);
    pub const LAYER_NORM_LAST_DIM: FusedOpId = FusedOpId(3);
    // ...
}
```

Rule code then matches `Op::Fused(FusedOps::SOFTMAX_LAST_DIM, _)` — almost as ergonomic as today's `Op::SoftmaxLastDim`. The constants are kept in sync with the registry initialization code via a build-time check (or a single source-of-truth macro that emits both).

### Q3: Can `BackendImpl` be `'static` to avoid Vec-of-trait-objects allocation?

The registry holds `BackendImpl` values; they're function pointers + small-struct fields. Should compose as plain structs without trait-objects — keeps the registry's storage flat. Vec<(BackendId, BackendImpl)> per entry; SmallVec for inline-up-to-4 backends.

### Q4: Does `register_fused!` macro live in fuel-graph or fuel-storage?

The macro spans both crates (it consumes registry-side metadata + binding-table-side kernel + `BackendImpl`). Probably lives in fuel-storage (the side that owns `KernelRef` and where existing `register_*_kernels` functions live), with re-exports through fuel-graph for ergonomics.

### Q5: How do CUDA-only (or backend-specific) fused ops work?

A fused op may have only one backend with a kernel for it. Architecturally fine: the registry entry has one `BackendImpl` populated; other backends fall back to the entry's `decompose` function (executing the primitive subgraph on a backend that doesn't have the fused kernel). Cost reflects this: the optimizer compares fused-on-CUDA vs decomposed-on-Vulkan honestly.

The registry doesn't need a "scope" concept (private vs shared FusedOpIds) until 50+ backend-specific fusions exist. Defer.

---

## Out of scope (this work item)

- **Cost-based scheduler implementation.** This refactor produces the substrate; the scheduler is downstream.
- **Multi-level dialect IR (MLIR-style).** Two layers — primitive Op variants + fused-op registry behind `Op::Fused` — is enough.
- **Pattern-match autotuning / e-graph equality saturation.** Anchored structural matching (PR 3 + declarative patterns) is sufficient. e-graphs as offline rule-discovery tool is future work.
- **User-extensible fused ops at runtime.** Registry frozen at startup; hot-add isn't a goal.
- **Bool dtype.** Comparison-op output is float (1.0/0.0) per the comparison-family decision in step 10. Bool dtype is independent and orthogonal.

---

## Honest caveats

This refactor touches the deepest layer of fuel — every executor, every backend, every autograd path matches on `Op`. Those matches all change shape. Mitigation: parallel-introduction-then-drop — existing variants and the new `Op::Fused` arm coexist throughout the migration window; per-fused-op variants drop in step 5. Each fused-op migration in step 4 is independently shippable.

The architecture's pre-resolved KernelRef commitment (step 9) is a meaningful refactor on its own — it changes where the binding table is consulted (planning time, not execution time). Lands in this phase because Phase 7.6's executor work is the natural place to also restructure the executor's per-node dispatch path.

PR 3's hand-written rules are easier to read than auto-generated rules from registry entries. Debugging a misbehaving fusion requires understanding the rule generator, not just the rule. Mitigation: keep the hand-written form available as escape hatch for canonicalization passes outside the auto-generation pattern; expose the rule generator's intermediate output for debugging.

Cost estimates (step 8) can mislead a scheduler. A FLOP-counting model misses fixed launch overhead, queue-wait latency, bandwidth interactions. Mitigation: cost estimates are advisory; the cost-aware scheduler also measures and adapts (Phase 6b empirical Judge feeds cost-model layer 2). Initial estimates can be coarse.

This phase should not run concurrently with Phase 8 (FlashAttention) or Phase 8.5 (sparsity); both add new fused ops mid-flight that would have to absorb the registry refactor. Phase 7.5 work items B/C/E are orthogonal.

---

## Success criteria

End-state criteria for the full Phase 7.6 (steps 1-11). Steps 1-3 met the subset already (marked ✓); the rest land as later sessions ship the remaining steps.

- ✓ `Op` enum carries `Op::Fused(FusedOpId, FusedOpParams)` arm (step 2). *Pending step 5*: drop the per-fused-op `Op` variants once nothing emits them; ~85 primitive variants remain.
- *Pending step 4*: `FusedOpRegistry` populated with 13-14 entries. Step 3 ships exactly one entry (SoftmaxLastDim) as proof-of-concept.
- ✓ PR 3's hand-written SoftmaxLastDim rules deleted; auto-generated `LoweringRule` + `FusionRule` from the registry entry produce equivalent behavior. Round-trip identity test (`softmax_last_dim_lower_then_fuse_round_trips`) still passes (step 3).
- ✓ Live CUDA equivalence test `cuda_executor_matches_cpu_on_softmax_via_lowering` passes through the registry-dispatched lowered subgraph (step 3; max abs err `4.47e-8` vs `1e-5` tolerance).
- *Pending steps 6-7*: every registered kernel carries a `PrecisionGuarantee`; the always-built backend's coverage commitment (one `bit_stable_on_same_hardware: true` kernel per primitive op) is testable as a CI lint.
- ✓ All existing tests green throughout the migration. CSE / op_key handles `Op::Fused(id, params)` correctly via `FusedOpParamsKey` encoding (step 2).
- *Pending step 11*: ROADMAP updated post-migration.

---

## References

- **Architecture v1.0**: [`docs/architecture/`](architecture/00-index.md). Sections 03 (IR), 04 (optimization), 05 (backend contract), 11 (persistence) are the most relevant.
- **PR 3 rule registry**: `fuel-graph/src/opt.rs` (`Rule`, `RuleFamily`, `RuleRegistry`) — substrate this refactor builds on.
- **Architecture audit**: `docs/architecture-audit.md` — the cross-thread audit that triggered architecture v1.0 drafting; surfaced Q-A (binding-table layer) which v1.0 resolved as planning-time pre-resolution.
- **Stablehlo op set**: `https://github.com/openxla/stablehlo` — reference for primitive-op-set sizing.
