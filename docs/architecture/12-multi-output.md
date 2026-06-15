# Multi-output nodes

**Status**: v1.0 (2026-06-01). Captures the Option-C design that lets a single graph node produce N logically independent outputs without growing the `DynBackendStorage` trait surface or forcing backends to learn "multi-output kernel" semantics. Landed across 6 sessions; the consumer migrations and the autograd scatter kernel light up downstream.

A multi-output node produces a bundled `Storage` — one allocation, one Arc, but a side-table describing how the byte buffer is partitioned into per-slot windows. Downstream `Op::View` (zero-copy projection) and `Op::ViewOwned` (independent slot copy) ops resolve those windows back into ordinary tensors. The bundle metadata lives on the `Storage` newtype; backends remain single-output at the trait level.

---

## Design choice

Three options were considered. **Option C** (bundled storage) was chosen.

### Option B (rejected): N independent storages

A multi-output node has N output slots, each with its own `Storage` allocation. Pro: independent per-slot liveness for free. Con: N allocator hits per multi-output op, compounding across thousands of fused ops per training step.

### Option C (chosen): bundled storage + per-slot views

A multi-output node allocates one bundled `Storage`. The bundle's bytes hold every slot's data contiguously at known offsets. Per-slot independent liveness is recoverable via `Op::ViewOwned` (small memcpy into a fresh standalone Storage), which the planner promotes from `Op::View` when a slot outlives the bundle.

**Why C over B**: backend trait surface stays single-output (`DynBackendStorage::dtype_dyn() -> DType` doesn't change); one allocator hit per multi-output op; fusion stays graph-rewrite-friendly because View projections are metadata a rule can elide.

### Option B' (rejected): bundled storage but `dyn DynBackendStorage` is the bundle

The bundle is built into the backend storage trait. Rejected: every backend (CPU, CUDA, Vulkan, MKL, AOCL, Metal) would need to learn the bundle vocabulary. Bundle metadata at the `Storage` newtype level keeps every backend impl untouched.

---

## The substrate

### `OutputView`

Per-slot metadata. Lives in `fuel_core_types::storage`:

```rust
pub struct OutputView {
    pub byte_offset:  usize,            // where this slot starts in the bundle's bytes
    pub len_elements: usize,            // count in the slot's dtype
    pub dtype:        DType,            // slot's dtype (NOT the bundle's primary)
    pub shape:        Shape,            // slot's logical shape
    pub layout:       Layout,           // slot's logical layout
    pub name:         Option<&'static str>, // optional debugging name
}
```

Slots are independent in dtype, shape, and layout. Two slots in the same bundle may carry different dtypes (e.g. F32 `y` + I64 `argmax_idx`).

### `OutputViewSpec`

Author-side per-slot spec — the `OutputView` minus the byte_offset/len_elements, which the allocator computes:

```rust
pub struct OutputViewSpec {
    pub dtype:  DType,
    pub shape:  Shape,
    pub layout: Layout,
    pub name:   Option<&'static str>,
}
```

`compose_bundle(&[OutputViewSpec]) -> Result<(usize /* total_bytes */, Vec<OutputView>)>` aligns each slot's `byte_offset` to its dtype size and emits the byte-offset-resolved `OutputView` list plus the total bundle byte count.

### `Storage` (both substrates)

`fuel_core_types::Storage` and `fuel_storage::Storage` each grew an optional `bundle: Option<Arc<[OutputView]>>` field. `None` for single-output (today's default); `Some(_)` when a multi-output op attached the side-table at construction time. The primary dtype (the inner backend storage's `dtype_dyn()`, slot 0's dtype) is enforced equal to `bundle[0].dtype` at attachment time.

### `allocate_bundled_storage`

`allocate_bundled_storage(device, &[OutputViewSpec]) -> Result<Storage>` calls `compose_bundle`, computes a flat element count in the primary dtype's units (rounded up), allocates via `device.zeros_impl_dyn(...)`, and attaches the bundle. Single call per multi-output op realization.

---

## The IR

### `Op::View { slot: u32 }`

Zero-copy projection. At realize time the executor clones the producer's `Arc<RwLock<Storage>>` into the View's cache slot. The View's `Layout` (set by `Tensor::view` at build time) bakes the slot's `byte_offset / dtype.size_in_bytes()` into `Layout::start_offset` so a downstream kernel reading `as_slice::<T>()[start_offset..]` lands on the slot's first byte.

### `Op::ViewOwned { slot: u32 }`

Independent slot allocation. At realize time the executor allocates a fresh contiguous Storage of the slot's `(shape, dtype)` and memcpys `producer.bytes[byte_offset..byte_offset+len_bytes]` into it. The producer's bundle Arc can drop independently once the View consumers all finish.

### `Op::ScatterIntoSlot { slot: u32 }`

Autograd assembler. Takes `[bundle_target, slot_grad]` and produces a copy of `bundle_target` with slot `slot`'s byte range overlaid by `slot_grad`. Emitted by `Op::View` / `Op::ViewOwned` backward rules to compose per-slot gradients into a bundle gradient for the producer. **IR-only today**: no kernel registered — production multi-output producers are `BackwardKind::NotDifferentiable`, so autograd panics at the producer before reaching the scatter. Lights up alongside the first differentiable multi-output op.

---

## The authoring contract

`FusedOpEntry::output_views: Option<fn(&[Shape], &[DType], &FusedOpParams) -> Vec<OutputViewSpec>>`.

- `None` for single-output ops (the default).
- `Some(fn)` for multi-output ops. Slot 0's spec MUST equal `shape_rule(...)` / `dtype_rule(...)` — slot 0 is the primary, and the producer's `Node::shape` / `Node::dtype` reflects it. The `default_registry_multi_output_entries` + `multi_output_entries_slot_0_matches_primary_rules` tests enforce this for every registered multi-output entry.

The `Tensor::*` builder for a multi-output op (`Tensor::selective_scan`, etc.) calls `entry.output_views(...)`, runs it through `compose_bundle`, and registers the result via `Graph::set_output_views(producer_id, views)` at build time.

### Bundled-tuple builders

`Tensor::selective_scan_bundled(...) -> Result<(Tensor, Tensor)>` returns both slots via `Op::View(0)` and `Op::View(1)` projections. `LazyTensor::selective_scan_bundled` is the higher-level wrapper. Mirror for `ssd_chunk_scan_bundled`.

---

## The graph side-table

`Graph` carries:

- `node_output_views: HashMap<NodeId, Arc<[OutputView]>>` — set via `Graph::set_output_views`, read via `Graph::output_views` / `output_views_arc`. Coherence rules enforced at set time:
  - non-empty slot list,
  - `views[0].dtype == nodes[id].dtype`,
  - `views[0].shape == nodes[id].shape`,
  - `views[i].layout.shape() == views[i].shape` for every `i`.
- `is_multi_output(id) -> bool` / `multi_output_count() -> usize` for diagnostics.

---

## Realization

### Bundled output allocation

In `fuel-dispatch::pipelined::compile_one`, the Kernel arm reads `graph.output_views_arc(id)` and attaches it to the `WorkItem` as `output_bundle: Option<Arc<[OutputView]>>`. At execute time, when `output_bundle` is `Some(_)`, the executor computes the total bundle byte span, allocates a Storage sized to hold every slot (on CPU, CUDA, or Vulkan via the existing per-backend allocators), and calls `Storage::with_bundle` to attach the metadata before the kernel runs.

### Kernel signature

Multi-output kernels emit ONE `KernelRef` with `outputs.len() == 1`. The single output is the bundled Storage; the kernel reads `outputs[0].bundle()` for per-slot specs and writes each slot's bytes via `split_at_mut` on a typed view, or by byte-offset arithmetic for mixed-dtype bundles. SelectiveScan + SsdChunkScan are the first authors of this contract.

### `Op::View` arm (`WorkItemKind::SlotView`)

Structurally identical to a single-input view-op realization: clone the producer's Storage Arc into the View's cache slot, and publish the View's layout from the WorkItem. The layout was prepared at build time with the slot's `byte_offset` baked into `start_offset`.

### `Op::ViewOwned` arm (`WorkItemKind::SlotOwn`)

Read the producer's bundled Storage's bytes via `to_host_buffer_dyn()` (CPU path), slice the slot's window, allocate fresh on the producer's device, and copy. Non-CPU backends return a typed error today; the per-backend `copy-with-offset` hook is the followup-prompt's deferred scope.

---

## Ordering

`opt::collect_alias_set` treats `Op::View` as alias-extending (the View shares the producer's bundle Arc, so destructive ops on the producer must run after every reader of every View). When the destructive root is itself an `Op::View`, the producer is pre-seeded into the alias set so sibling Views are reached via the forward walk. `Op::ViewOwned` is NOT alias-extending — the memcpy produces an independent Storage; the regular data-dependency edge (`inputs[0] == producer`) covers the ordering.

`Op::Release` on a bundled producer is documented as a single-eviction-unit operation: the whole bundle drops when the last View consumer finishes. Per-slot eviction is intentionally a follow-up.

---

## Planner

`opt::promote_views_for_liveness` walks each multi-output producer, computes per-slot last-use depth via the forward `depth[N]` graph (longest input chain feeding into N), and promotes `Op::View` → `Op::ViewOwned` for slots whose downstream depth exceeds at least one sibling slot's depth. The classic case: `y` consumed at the next layer, `last_state` retained across an autoregressive barrier. Idempotent on `Op::ViewOwned`.

---

## What's deliberately deferred

- **`Op::ScatterIntoSlot` kernel.** Needs a differentiable multi-output producer to be load-bearing. Mamba training is the obvious trigger; FSCE loss+grad bundling is another.
- **Per-backend `Op::ViewOwned` copy-with-offset hooks** for CUDA / Vulkan. The CPU path ships; GPU follow-up is the consumer-migration session's scope.
- **`Op::Copy` of a bundled producer.** Today errors cleanly; whole-bundle cross-device copy needs a per-backend bundle-aware Copy hook.
- **Mamba consumer migration.** Eager `fuel_core::Tensor` → `LazyTensor` path. Once the eager-to-lazy session lands, Mamba's inference loop uses `selective_scan_bundled` for resumption.
- **FSCE loss+grad bundling.** Same pattern as SelectiveScan migration; its own session.

---

## Cross-references

- IR scaffolding: [`03-ir.md`](03-ir.md) — multi-output ops live in the `Op::Fused` arm; the registry's `output_views` field is the multi-output authoring contract.
- Backend contract: [`05-backend-contract.md`](05-backend-contract.md) — backends remain single-output at the trait level; bundle metadata is at the `Storage` newtype layer.
- Runtime: [`06-runtime.md`](06-runtime.md) — `WorkItemKind::SlotView` / `SlotOwn` arms in the pipelined executor; the alias-aware ordering pass treats `Op::View` as bundle-aliasing.
- Session prompts: [`docs/session-prompts/shipped/multi-output-nodes-option-c.md`](../session-prompts/shipped/multi-output-nodes-option-c.md) for the original design rationale; [`docs/session-prompts/shipped/selective-scan-ssd-chunk-multi-output-followup.md`](../session-prompts/shipped/selective-scan-ssd-chunk-multi-output-followup.md) for the consumer migrations.
