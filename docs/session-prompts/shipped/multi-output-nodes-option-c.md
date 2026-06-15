# Session prompt — Multi-output nodes via bundled storage (Option C)

## What this session is for

Land the infrastructure that lets a single graph node produce multiple
logically independent outputs — each with its own dtype, shape, and
layout — without changing the `DynBackendStorage` trait surface or
forcing backends to learn "multi-output kernel" semantics.

The chosen design (call it **Option C**) is: a multi-output op
allocates **one bundled `Storage`**, and a side-table on that Storage
enumerates per-output views (`offset`, `len`, `dtype`, `shape`,
`stride`). Downstream consumers reference outputs via two new graph
ops — `Op::View` (metadata-only, zero copy) and `Op::ViewOwned`
(memcpy of just that slice into its own Storage). Cross-device
movement is the existing `Op::Copy`, no new primitive needed.

The motivating consumers are SelectiveScan + SsdChunkScan with their
`last_state` outputs (Mamba autoregressive resumption), and FSCE-style
loss-and-grad bundles. Multi-output infrastructure should land
**before** those consumers migrate, so they can be built against the
final shape from day one.

## Design decisions already settled (do not re-litigate without strong evidence)

These were debated in the design session that produced this prompt. If
you want to revisit any, raise it explicitly — don't quietly diverge.

### 1. Bundled storage (C), not multi-output node (B)

A multi-output node could also be expressed as N independent Storage
allocations behind a `Node` that has N outputs (Option B). We chose
**C** because:

- **Backend trait surface stays single-output.** `DynBackendStorage::dtype_dyn() -> DType`
  doesn't change. Every backend (fuel-cpu-backend, fuel-cuda-backend,
  baracuda, fuel-vulkan-kernels, fuel-mkl, fuel-aocl,
  fuel-metal-backend) is untouched at the trait level.
- **Byte substrate carries it natively.** `CpuStorageBytes` is already
  `Arc<AlignedBytes>` — dtype-agnostic. The substrate doesn't care
  whether a buffer holds one dtype or several at different offsets;
  typed access goes through `bytemuck::cast_slice::<T>()` either way.
  See [fuel-cpu-backend/src/byte_storage.rs:1-15](../../fuel-cpu-backend/src/byte_storage.rs#L1-L15).
- **One allocator hit per multi-output op.** B would do N. Compounds
  across thousands of fused ops per training step.
- **Fusion stays graph-rewrite-friendly.** `Op::View` projections are
  metadata that a fusion rule can elide without the producer learning
  multi-output semantics.

The single semantic advantage B has — independent sub-output liveness —
is recoverable in C via `Op::ViewOwned` (see below).

### 2. Three new ops, not one fused "ProjectTo"

We rejected a hypothetical `Op::ProjectTo { node, index, target_device }`
that would have fused View + Copy into one op. Reasons:

- No kernel-level win — Copy already takes a Storage and can read from
  any offset; the View input gives it the exact info ProjectTo would
  carry.
- No DMA path that View+Copy can't already hit.
- Composition, not new semantics. The rule for adding to the Op enum:
  add when the IR can't express the thing otherwise. ProjectTo fails
  that test.
- Graph optimizer can fuse the View→Copy pattern at lowering time if
  profiling ever shows it matters — that's what the rule registry is
  for.

What we DO want is a builder-side ergonomic helper:
`lazy_tensor.project_to(view_idx, device)` that emits `View + Copy`
under the hood. Not an Op.

### 3. `Op::View` and `Op::ViewOwned` are distinct

These are NOT the same op with a flag:

- **`Op::View { node, index }`** — clones the bundle's `Arc`, exposes
  the slot's dtype/shape/offset. Zero bytes moved. **The bundle stays
  alive as long as any View (or the original bundle handle) holds the
  Arc.** Default choice.
- **`Op::ViewOwned { node, index }`** — memcpy of just that slot's
  bytes into a fresh standalone `Storage`. Costs one allocation + one
  slot-sized copy. **The bundle can drop as soon as the last View
  releases its Arc.**

The planner picks View vs ViewOwned based on liveness analysis:
default to View, switch to ViewOwned when keeping the bundle alive
costs meaningfully more bytes than the ViewOwned memcpy would.

### 4. Cross-device works trivially

The producer-device locality is not a single-device-only constraint —
it's "the bundle lives on its producer's device until something
explicitly moves bytes." Two patterns:

- Whole-bundle move: `Op::Copy(bundle, target_device)` — moves the
  whole multi-output buffer, then `View` ops on the new device project
  per-output. Useful when most or all outputs are headed to the same
  destination.
- Per-slot move: `Op::View → Op::Copy → Op::ViewOwned` (or just
  `Op::View → Op::Copy` if the bundle drops naturally). Useful when
  one output goes to GPU and another stays on CPU.

## Concrete deliverables

### 1. Storage side-table for output views

In `fuel-core-types/src/storage.rs` (or a sibling module), add an
optional bundle-metadata field to `Storage`. Suggested shape:

```rust
pub struct OutputView {
    pub byte_offset: usize,
    pub len_elements: usize,
    pub dtype: DType,
    pub shape: Shape,
    pub layout: Layout,   // own stride/contiguity, independent per slot
    pub name: Option<&'static str>,   // optional, debugging
}

pub struct Storage {
    inner: Box<dyn DynBackendStorage>,
    // None == today's single-output behavior; dtype/shape via inner.
    // Some(_) == multi-output bundle; inner.dtype_dyn() returns the
    // "primary" dtype (typically the first slot) but real per-slot
    // info comes from this side-table.
    bundle: Option<Arc<[OutputView]>>,
}
```

Constraints:

- `bundle` lives on `Storage`, not on `DynBackendStorage`. The trait
  stays as-is. Backends produce a regular `Box<dyn DynBackendStorage>`;
  the bundle metadata is attached at the `Storage` newtype level when
  a multi-output op constructs the result.
- `dtype_dyn()` on a bundled storage returns something sensible (first
  slot's dtype, or a sentinel — pick deliberately and document). Most
  call paths going through `Storage::dtype()` should be rerouted to
  `Storage::primary_dtype()` or `Storage::slot_dtype(idx)` so the
  semantics are explicit at the call site.
- `same_dtype()`, the binding-table key, and dispatch-key construction
  on the consumer side ALWAYS operate on a projected view's dtype, not
  on the bundle's primary. Audit every call site.

### 2. Two new graph ops

In `fuel-graph/src/lib.rs` and `fuel-graph/src/op.rs`:

- `Op::View { source: NodeId, slot: u32 }` — node output is a Storage
  that shares the source bundle's `Arc` and exposes one slot's dtype +
  shape + layout. No backend dispatch — handled at graph realization
  by reading the source's `bundle[slot]` and constructing a Storage
  whose `dtype_dyn` reports the slot's dtype.
- `Op::ViewOwned { source: NodeId, slot: u32 }` — node output is a
  freshly allocated Storage of the slot's dtype/shape, populated by a
  memcpy of `bundle.bytes[byte_offset .. byte_offset + len_bytes]`.
  This DOES dispatch (CPU memcpy, CUDA d2d, Vulkan d2d, …) but the
  per-backend impl is a one-liner — it's `Op::Copy` with offset+len.

Both ops need: shape inference (trivial — read from bundle metadata),
dtype inference (trivial), backward rules (the backward of a View is
"accumulate this slot's grad into a zero-filled bundle of the source's
shape" — see §4 below), and layout-on-node integration.

### 3. Multi-output op authoring contract

Codify how a fused-op author (e.g., the FusedOpEntry for SelectiveScan)
declares N outputs. Suggested in `fuel-graph/src/registry.rs`:

- `FusedOpEntry::output_views(&self, params: &FusedOpParams) -> Vec<OutputViewSpec>`
  — returns the per-slot dtype/shape derived from inputs + params.
  Validated at graph-build time (consistent with
  `validate-at-graph-build-time` feedback).
- The Storage allocator for a multi-output op computes the total byte
  count (sum of `slot.bytes` with per-slot alignment) and allocates
  one bundled Storage in one call.
- Kernel signature: each backend's wrapper for a multi-output op
  receives a single output Storage and the bundle metadata, then
  writes each slot via the appropriate typed offset (e.g.,
  `as_slice_mut::<f32>()[y_start..y_end]` for slot 0,
  `as_slice_mut::<f32>()[state_start..state_end]` for slot 1).

### 4. Autograd integration

Backward through a View is "scatter grad into the slot's offset of a
zero-filled bundle." Backward through a multi-output op then combines
the per-slot grad-bundles by summing them and feeding the combined
bundle to the op's existing backward rule (which already knew how to
take a bundle gradient — it produced one).

Concretely:

- `Op::View`'s backward rule emits `Op::ScatterIntoSlot { bundle_zero,
  slot, grad }` (or equivalent — a tiny per-slot scatter, not a full
  multi-output write).
- When N View consumers of the same source node all run their
  backward, the N scatters into a shared zero bundle compose
  associatively into the source op's gradient input.
- `Op::ViewOwned`'s backward is "scatter this grad into the bundle's
  slot offset" — same shape as View, just that the forward already
  paid a copy.

Decide explicitly: does the multi-output op's backward see a *single
bundle gradient* (clean — symmetric with forward) or a *slice of
per-slot grads* (potentially saves a zeroed allocation but uglier
contract)? Recommendation: bundle gradient, for symmetry.

### 5. Planner liveness for View vs ViewOwned

Add a planner pass that walks each multi-output node's downstream
consumers and decides per slot:

- If all the bundle's slots have similar lifetimes (consumed within
  the same "phase" of the graph): use `View` for all of them. Bundle
  drops naturally when last consumer is done.
- If one slot is short-lived and another is long-lived (e.g.,
  `y` consumed immediately, `last_state` retained for next step):
  promote the long-lived one to `ViewOwned`. Short-lived stays
  `View`. The bundle drops when the short-lived consumer finishes;
  the long-lived's ViewOwned-copied Storage stands on its own.

This pass is the C-design's analog of B's "independent liveness for
free" — same end state, paid via a small memcpy rather than a separate
allocation per output. Keep it as a lowering pass, not part of the
op author's burden.

### 6. Scheduler / destructive-cleanup integration

`derive_ordering` and `insert_safety_copies` already track per-Storage
liveness. They need to extend to bundle-aware liveness:

- A bundle is alive while any `View` referencing it is alive (Arc
  refcount on the bundle's underlying storage).
- `ViewOwned` ops finalize their input dependency at execution time
  (after the memcpy runs, the View input can drop).
- The bundle is a single eviction unit — when the residency pass
  decides to evict, it evicts the whole bundle. (If finer-grained
  eviction matters, that's a follow-up; not v1 scope.)

## What's NOT in scope for this session

- **Migrating SelectiveScan / SsdChunkScan to actually use multi-output.**
  That's the consumer-side session (`selective-scan-ssd-chunk-multi-output-followup.md`).
  This session ships the infrastructure; that session lights it up.
- **Migrating FSCE to multi-output for loss+grad bundles.** Same reasoning.
- **Multi-output for already-shipped ops.** Don't retrofit anything; just
  build the infrastructure.
- **The fusion-rule that collapses `Op::View → Op::Copy` into a single
  backend call.** Add only if profiling shows it matters.
- **`Op::ScatterIntoSlot` as a separate primitive.** Probably not
  needed — express via existing scatter / copy_strided primitives.
  Decide during implementation.

## Scope estimate

This is genuinely 2-3 focused sessions, not 1:

- **Session 1** — Storage side-table + `Op::View` + `Op::ViewOwned` +
  shape/dtype/layout inference + autograd for views. No multi-output
  op authoring yet; tests use a hand-rolled bundled-storage builder
  to exercise View / ViewOwned semantics in isolation. ~1 day.
- **Session 2** — Multi-output op authoring contract on
  `FusedOpEntry` + the bundled allocator + the planner liveness pass
  for View vs ViewOwned. Still no real consumer — tests use a
  synthetic 2-output fused op. ~0.5-1 day.
- **Session 3** — Scheduler / destructive-cleanup integration +
  bundle-aware residency. ~0.5 day.

Then the consumer migration sessions can start.

If you want to ship in fewer sessions, fold Session 2 + 3 together —
they share the same code paths and the test surface for both is the
same synthetic-op fixture.

## Coordination

- **Eager-to-lazy migration session.** Mamba is migrating eager →
  lazy in parallel. Once multi-output infra lands, the
  SelectiveScan/SsdChunkScan `last_state` work depends on it; loop
  the Mamba migration session in so they don't redo the same plumbing
  via a one-off escape hatch.
- **Picker / planner session.** The View-vs-ViewOwned planner pass
  conceptually lives in the same area as the alternative-set picker
  work. Coordinate with that session on where the pass plugs in
  (probably right after ranker, before `compile_plan`).
- **Dispatch crate.** The View and ViewOwned ops live in fuel-graph;
  dispatch wrappers (for ViewOwned only — View is metadata) live in
  fuel-dispatch. Mirror the pattern of other "trivial" ops like
  Op::Copy.

## References

- Design discussion: this session's predecessor context (filename
  redacted — the conversation summary that produced this prompt).
- Memory: `project_cpu_opkind_followups_shipped.md` — context on the
  SelectiveScan/SsdChunkScan `return_state` attempt that motivated
  this work.
- Memory: `feedback_architectural_cleanness_over_local_pragmatism.md`
  — the principle that drove choosing C over a sibling-op shortcut.
- Memory: `feedback_validate_at_graph_build_time.md` — apply to the
  bundled-allocator shape/dtype derivation.
- Architecture: `docs/architecture/` — the Storage / Op / Node
  layering doc set.
- Existing single-output `Storage` definition:
  [fuel-core-types/src/storage.rs](../../fuel-core-types/src/storage.rs).
- Byte substrate proof: [fuel-cpu-backend/src/byte_storage.rs](../../fuel-cpu-backend/src/byte_storage.rs).
- `DynBackendStorage` trait (must stay unchanged):
  [fuel-core-types/src/dyn_backend.rs](../../fuel-core-types/src/dyn_backend.rs).
