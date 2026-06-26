# fuel-core / fuel-core-types retirement — B0 program

**Status (2026-06-26):** B0.1 SHIPPED (`f7e7af43`). B0.2a SHIPPED (`a78801e8` — fuel-hardware
crate + `HardwareEnumerator` registry + probe moved). B0.2b SHIPPED (`ecef5e96` — transfer_cost
moved). **Remaining: B0.2c (topology split + relocate SystemTopology overlay to fuel-dispatch +
clean factories.rs's now-dead `enumerate_devices` + rewire the 2 pipelined_bridge.rs sites), then
B0.3–B0.5.** This doc is the resume artifact — it captures the code-grounded investigation (a
4-agent sweep) so the remaining steps don't need re-deriving.

## Why

`fuel-core` and `fuel-core-types` both collide with existing crates.io names and must be
retired before publish (memory `[[fuel-core-retirement]]`). The user's directive
(2026-06-26): do the **full untangle** — get every type to its true home, not just a rename.

## The investigation finding — `fuel-core-types` was a GRAB-BAG

Not a cohesive vocabulary crate; four concerns accreted under one roof (it's the bottom
crate, so everything could be parked there to dodge dependency cycles):

1. **Vocabulary** (the real IR labels/values/identities): `dtype` (DType/WithDType),
   `shape` (Shape/Extent/DynAxis — incl. Phase-D symbolic extents), `layout`, `scalar`,
   `symbol` (SymId/SymEnv/DynScalar), `device` (DeviceLocation), `op` (Bin/Un/Cmp/Reduce),
   `conv` params, `quant_scale` (ScaleGranularity/ScalePair), `dummy_dtype` (F4/F6/F8 tags),
   `dispatch::OpKind`, `probe::BackendId`, `capability`, `error`, `strided_index`,
   `backend::{SubstrateClass, TransferPath}` (these two are vocabulary, used by fuel-dispatch).
2. **Storage-impl** (real bytes — belongs in fuel-memory per `docs/foundational-types.md`):
   `storage.rs` (old typed `Storage` = Box<dyn DynBackendStorage> + eager-dispatch surface;
   consumed by fuel-graph's storage_map; overlaps fuel-memory's newer byte-Storage —
   fuel-memory even imports its `OutputView`), `cpu_storage.rs` (`HostBuffer` = owned Vec<T>,
   the cross-backend host-interchange buffer — pervasively imported), the `cpu/` SIMD+erf
   kernels (`VecOps`), `quantized.rs` impl traits, `inplace_op.rs`.
3. **Backend-contract / discovery**: `dyn_backend` (DynBackendStorage/DynBackendDevice —
   every backend implements), `backend.rs` (BackendStorage/BackendRuntime/BackendCapabilities/
   HostStorage/FitStatus), `quantized` Dyn* traits, `probe` (BackendProbe/DeviceDescriptor/
   EquivalenceKey discovery), `capability`.
4. **Dispatch overlay** (stays in fuel-dispatch): most of `dispatch.rs` — ProfileReport/
   DispatchTable/Pick/SizeClass (only OpKind is vocabulary).

### Why a clean "fuel-ir = vocabulary only" split was NOT a free rename (the two hard welds)
- **`WithDType` is welded to storage-impl.** `pub trait WithDType: ... + crate::cpu::kernels::VecOps`
  (a supertrait bound) AND method signatures returning/consuming `HostBuffer`/`HostBufferRef`
  (cpu_storage). So `DType` cannot leave for a pure-vocab crate without dragging `cpu/` +
  `cpu_storage` along, OR redesigning `WithDType` (touches every backend's impl). `dummy_dtype`
  also `impl ... VecOps`.
- **`error.rs` leaks `backend::TransferPath`** into an `Error` variant (a 2nd vocab→nonvocab edge).

This is why B0 is sequenced (below), not one big-bang.

## The 5-step decomposition (each its own commit + per-crate verify; one cargo at a time)

- **B0.1 — Rename fuel-core-types → fuel-ir. SHIPPED `f7e7af43`.** Wholesale `git mv` +
  sed (`\bfuel_core_types\b`→`fuel_ir`, `fuel-core-types`→`fuel-ir`). 409 rs / 17 toml, 0
  stragglers. fuel-core's granular re-exports insulate downstream `fuel_core::*`. Verified:
  check (fuel-ir/memory/graph/dispatch/core/cpu-backend) + 287 doctests. (3 PRE-EXISTING
  doctest failures: lazy_bert/lazy_convnext/lazy_sd_text_encoder call `.realize_f32()` on
  `forward()`'s Result without `?` — unrelated to the rename; needs a separate fix.)

- **B0.2 — Extract `fuel-hardware`** (discovery). The one clean extraction; serves the named
  goal. Plan (from the investigation):
  1. Create `fuel-hardware` crate. Deps: `fuel-ir` + `fuel-cpu-backend` + (cfg) `fuel-cuda-backend`
     + (cfg) `fuel-vulkan-backend` + serde. No cycle (backends' only fuel-core dep is dev-only in vulkan).
  2. Move `fuel-core/src/probe.rs` → fuel-hardware; replace its `crate::factories::registry()`
     call with a hardware-local enumerator registry (slim trait `{ id(), enumerate_devices() }`
     over CPU/Cuda/Vulkan, each delegating to the backend crate's `probe::enumerate_devices` free fn).
  3. Move `fuel-core/src/transfer_cost.rs` whole (TransferEstimate/TransferCalibration have NO
     cross-crate consumers besides SystemTopology — relocate cleanly).
  4. Split `fuel-core/src/topology.rs` along discovery/overlay: discovery (devices, transfer_paths,
     transfer_calibration, probe-derived build, default_*) → fuel-hardware; overlay (capabilities
     from `global_registry()`, binding_op_coverage from `global_bindings()`, the
     `topology_generation` counter, the `TransferEstimator` impl) → **fuel-dispatch**. RECOMMENDED:
     move the `SystemTopology` struct itself to fuel-dispatch (every overlay input it reads already
     lives there) and have it call into fuel-hardware for the pure-discovery pieces. Dep direction:
     fuel-dispatch → fuel-hardware → fuel-ir (fuel-hardware must NOT depend on fuel-dispatch).
  5. Split `BackendFactory` (`fuel-core/src/factories.rs`): `HardwareEnumerator` (`id()` +
     `enumerate_devices()`) → fuel-hardware; `RealizerFactory` (`try_make_realizer` + `LazyRealizer`
     + `BridgeRealizer`, which use LazyTensor/pipelined_bridge/StorageCache) STAYS in fuel-core.
     `judge.rs` is the sole `try_make_realizer` consumer (stays).
  6. Rewire the **2 production `SystemTopology` call sites** in `fuel-core/src/pipelined_bridge.rs`
     (build_optimized_graph ~L329; insert-copies/shares_storage ~L1406) to wherever it lands.
     No fuel-dispatch code edits (the `TransferEstimator` trait + closure params mediate; the
     `SubstrateClass`/`TransferPath` vocab stays in fuel-ir).
  7. Verify: `cargo check -p fuel-hardware -p fuel-dispatch -p fuel-core`; topology/transfer_cost/
     probe unit tests at the new home; live CUDA/Vulkan calibration tests one suite at a time.

- **B0.3 — Cut a backend-contract crate** (e.g. `fuel-backend-contract`): move `dyn_backend`
  (DynBackendStorage/DynBackendDevice), `backend.rs` (BackendStorage/BackendRuntime/
  BackendCapabilities/HostStorage/FitStatus), the quantized `Dyn*` traits, `inplace_op` OUT of
  fuel-ir into it. Every backend depends on it (so it sits below the backends, above fuel-ir).
  Break the `error.rs`→`TransferPath` leak first (or keep TransferPath in fuel-ir, which is fine —
  it's vocabulary). Heaviest verify (every backend).

- **B0.4 — Break the `WithDType` ↔ `cpu/`/`cpu_storage` weld.** Redesign `WithDType` to drop the
  `VecOps` supertrait bound + the `HostBuffer`/`HostBufferRef` method signatures (move those
  concerns to a separate trait in fuel-memory or the backend-contract crate). Touches every
  backend's `WithDType` impl. Prerequisite for B0.5 moving cpu/cpu_storage out of fuel-ir.

- **B0.5 — Relocate storage-impl → fuel-memory.** `storage.rs` (old `Storage` + `OutputView` —
  merge with fuel-memory's byte-Storage; this is the Storage-unification, itself a sub-program),
  `cpu_storage` (HostBuffer), `cpu/` (VecOps SIMD), `quantized` buffer impl. Now feasible after B0.4.

## Verification crates (never workspace-wide — `tensor-tools` has a standing break)
`fuel-ir` (+ `--doc`), `fuel-memory`, `fuel-graph`, `fuel-dispatch`, `fuel-core` (+ `--doc` for
the granular re-exports & macro bodies in op.rs/tensor.rs), `fuel-cpu-backend`, `fuel-cuda-backend`
(heaviest macro user — live CUDA per environment-discipline DevShell incantations),
`fuel-vulkan-backend`, `fuel-inference`, `fuel-reference-backend --tests`.

## Pre-existing items surfaced (out of B0 scope, flag/fix separately)
- 3 broken fuel-core doctests (realize_f32-on-Result, above).
- `build_errors{,2,3,4}.txt` — tracked junk in repo root.
- `fuel-dispatch/src/fused.rs:1-15` module doc still says `fuel-storage::fused` (stale comment).
