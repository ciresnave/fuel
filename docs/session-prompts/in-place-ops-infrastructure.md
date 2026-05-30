# Session prompt — In-place ops infrastructure

## What this session is for

Add first-class in-place op support to Fuel's lazy graph, following
the graph-tracked-version-safety design captured in
[`project_graph_tracked_version_safety.md`](../../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_graph_tracked_version_safety.md).
End state: model code can call `tensor.relu_inplace()` (and
similar) to mutate in place, the lazy graph schedules the mutation
respecting destructive-input ordering, and the autograd
mutation-safety pass either accepts the graph or auto-inserts a
clone when an in-place op would destroy a tape-tracked tensor.

## What's already in place (not just speculation)

Investigated 2026-05-29:

- **`Op::destructive_input() -> Option<usize>`** at
  `fuel-graph/src/lib.rs:770`. Returns the input index a node
  destroys on execution. Currently only `Op::Release`, `Op::Move`,
  `Op::WriteSlice`, `Op::ZeroFill` are destructive.
- **`opt::derive_ordering()`** at `fuel-graph/src/opt.rs:1389`
  walks the topo-order, finds each destructive op, and emits
  "must-run-after-all-readers-of-destroyed-input" ordering edges.
  Tests at lines 1872-1924 prove it works for `Release`.
- **`execution_plan()`** (also in opt.rs) respects both data-flow
  edges and the ordering edges. Fast path: zero overhead when the
  graph has no destructive ops.
- **`baracuda` kernels mostly accept same-pointer src/dst** for
  the elementwise unary + binary families (kernels write index
  `i` after reading index `i`; no aliasing issues). A few
  explicit `inplace_*` symbols also exist:
  `affine_inplace_{f32,f64}`, `scale_inplace_{c32,c64,real_*}`,
  `loss_flce_inplace_scale_{f32,f16,bf16,f64}`.

The architectural scaffolding is **already there**. This session
extends it.

## Phasing

### Phase 1: Op IR variants + destructive_input marking

Add to `fuel-graph/src/lib.rs::Op`:

```rust
/// In-place unary op — mutates input 0 in place, output aliases
/// input 0. Backward requires the original input value for many
/// unary ops (relu needs sign(x), exp needs y itself, etc.) — the
/// mutation-safety pass (Phase 4) handles that.
InplaceUnary(UnaryOp),  // UnaryOp from fuel-graph reuses existing enum

/// In-place scalar-affine — `x = a * x + b`. Backward only needs
/// the upstream gradient, no saved input. Always safe under
/// autograd.
InplaceAffine { mul: f64, add: f64 },

/// In-place binary — `x op= y`, mutating input 0. Backward
/// gradient distribution: same upstream gradient flows to both
/// pre-mutation x and y for Add/Sub. Mul/Div need saved values.
InplaceBinary(BinaryOp),
```

For each, implement `destructive_input() -> Some(0)`. The
existing `derive_ordering` pass will pick them up automatically —
no new optimizer code needed for the basic scheduling.

Touch points for exhaustive matches (verified 2026-05-29 across
~20 files):

- `fuel-core/src/backprop.rs` — autograd, lines ~639 + ~149 + 603
- `fuel-cuda-backend/src/backend.rs` line 118, `dyn_impl.rs` 98
- `fuel-cuda-backend/src/lib.rs` lines 373 + 777 (dispatch + name)
- `fuel-graph/src/grad.rs` line 72 (gradient rules)
- `fuel-graph/src/lib.rs` lines 865 + 2380 + 4596 + 6841 (naming,
  builder, executor, tests)
- `fuel-graph/src/opt.rs` line 737 (cost model)
- `fuel-graph-cpu/src/backend.rs` 194, `lib.rs` 349
- `fuel-graph-executor/src/lib.rs` lines 1510 + 2102
- `fuel-metal-backend/src/dyn_impl.rs` 95
- `fuel-reference-backend/src/exec.rs` 373
- `fuel-storage/src/cast_fusion.rs` 101, `pipelined.rs` 942
- `fuel-core-types/src/dispatch.rs` (OpKind enum)

Most arms can be NOP-or-error initially; we wire dispatch in
Phase 3.

### Phase 2: User-facing Tensor methods

Add to `fuel-core/src/tensor.rs`:

```rust
impl Tensor {
    /// In-place ReLU. Mutates `self`'s storage. Returns the same
    /// tensor handle for chaining; subsequent reads see the
    /// mutated values via the shared `Arc<RwLock<Storage>>`.
    ///
    /// **Autograd:** if `self` is tape-tracked, the mutation-
    /// safety pass (Phase 4) auto-inserts a `Op::Clone` before the
    /// mutation and rewires backward consumers to read the
    /// pre-mutation value. Transparent to model code — there's no
    /// case where `relu_inplace()` panics or errors under
    /// autograd.
    pub fn relu_inplace(&self) -> Result<Tensor> {
        self.emit_inplace_unary(UnaryOp::Relu)
    }

    pub fn silu_inplace(&self) -> Result<Tensor> { ... }
    pub fn gelu_inplace(&self) -> Result<Tensor> { ... }
    pub fn scale_inplace(&self, scale: f64) -> Result<Tensor> { ... }
    pub fn affine_inplace(&self, mul: f64, add: f64) -> Result<Tensor> { ... }
    pub fn add_inplace(&self, other: &Tensor) -> Result<Tensor> { ... }
    pub fn sub_inplace(&self, other: &Tensor) -> Result<Tensor> { ... }
    pub fn mul_inplace(&self, other: &Tensor) -> Result<Tensor> { ... }
}
```

The `emit_inplace_*` helpers push the new Op variants onto the
graph and return a tensor handle whose `Tensor_::data` is the
same `Arc<RwLock<Storage>>` as `self.data`.

### Phase 3: Backend execution

CPU + CUDA + Metal need backend functions that perform the
mutation:

```rust
// CudaStorage::inplace_unary
pub fn inplace_unary(&mut self, kind: UnaryOp, layout: &Layout) -> Result<()> {
    // Most existing kernels accept src_ptr == dst_ptr; verify per kernel
    // and use a single-pointer call shape.
    let (contig_fn, strided_fn) = pick_unary_ffi(kind.kernel_name(), self.dtype())?;
    // ... call with x == y
}
```

baracuda kernels confirmed safe for same-pointer dispatch:
elementwise unary, binary, affine (which baracuda already has as
`affine_inplace_*`). Verify per-family in the actual session
work.

### Phase 4: Mutation-safety optimizer pass

New rule in the optimizer's pre-execution pipeline. Walks the
graph; for each `SavedForBackward(node_id=X)` edge (from the tape
records — see `fuel-core/src/backprop.rs` for tape construction
shape), check whether any `Op::Inplace*(target_idx=X)` appears
between the save point and the backward consumer. On conflict:

1. Insert `Op::Clone(X) -> X_backup` immediately before the
   `Op::Inplace*`.
2. Rewire the `SavedForBackward` edge to point to `X_backup`.
3. Let the `Op::Inplace*` proceed on `X` as the user wrote.

The graph optimizer's existing rule-registry pattern
([`graph_optimizer_architecture`](../../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_graph_optimizer_architecture.md))
is the right home for this rule.

Storage gains **zero** new runtime state across all 4 phases.

### Phase 5 (future): automatic in-place rewriting

When the optimizer sees `y = relu(x); z = next(y)` and `x` has
no other live consumers, it can compile to `relu_inplace(x); z =
next(x);`. Pure optimization — model code stays functional.
Save for after Phases 1-4 are proven in real model code.

## What NOT to do

- Don't add `_version: AtomicInt` to `Storage`. The design memo
  ([`project_graph_tracked_version_safety.md`](../../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_graph_tracked_version_safety.md))
  explicitly rules this out. Static graph analysis subsumes the
  runtime check.
- Don't expose `&mut Storage` methods to user model code. The
  only mutation path is through `Op::Inplace*` graph nodes.
- Don't introduce an "eager mode" gate. Fuel is always-graph
  post Phase 7.5; the user reminded us of this 2026-05-29.

## Test scope

- Smoke: `let y = x.relu_inplace()?` works when `x` is detached
  (no autograd tape). Storage values match `x.relu()?` output.
- Aliasing: `let y = x.relu_inplace()?; assert!(Arc::ptr_eq(...))` —
  `y` and `x` share storage.
- Scheduling: if `let y = x.relu()?; let z = x.relu_inplace()?;`
  builds a graph, `derive_ordering` ensures the inplace runs
  after the non-inplace read. Existing
  `derive_ordering_release_must_run_after_sibling_readers` test
  at `opt.rs:1881` is the pattern to mirror.
- Autograd safety (Phase 4): build a graph where `x` is saved
  for backward and a subsequent op tries to mutate it; assert the
  mutation-safety pass inserts a clone and backward still
  produces the right gradient.
- Regression: full `cargo test -p fuel-core --features cuda` sweep
  to confirm no existing behavior breaks.

## Scope realism

Phases 1-3 are 3-5 sessions of work depending on how thoroughly
the backend impls are wired across the 3 backends. Phase 4 is its
own session (heavy autograd integration). Phase 5 is a bonus.

The user has explicitly approved building this; the constraint is
correctness + not breaking existing behavior, not whether to do
the work.

Link:
[`project_graph_tracked_version_safety`](../../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_graph_tracked_version_safety.md),
[`project_graph_optimizer_architecture`](../../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_graph_optimizer_architecture.md).
