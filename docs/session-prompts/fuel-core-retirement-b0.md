# fuel-core / fuel-core-types retirement — B0 program

**Status (2026-06-26):** B0.1 SHIPPED (`f7e7af43` — rename). **B0.2 COMPLETE** — 2a (`a78801e8`:
fuel-hardware crate + `HardwareEnumerator` registry + probe), 2b (`ecef5e96`: transfer_cost), 2c
(`3a0b1d88`: SystemTopology relocated to fuel-dispatch, the overlay home). fuel-hardware now owns
hardware discovery; fuel-core owns none. **B0.4 host-method weld DONE** (`686554fc`: split
`WithDType` → `WithDType` + `HostDType`; the 5 host-buffer methods moved to `HostDType` in
cpu_storage.rs; fuel-core marker += HostDType; cuda/metal `storage_from_slice` += HostDType.
Verified fuel-ir+cpu-backend+fuel-core+cuda; metal untested-marked). The `VecOps` supertrait drop
was **deferred from B0.4 to B0.5** (it requires Map2-trait churn + is coupled to the cpu/ move).
**Remaining: B0.3 + B0.5** (below), plus the factories.rs dead-`enumerate_devices` cleanup (fold
into B0.3). NOTE on verification cost: a `fuel-cuda-backend` build is ~36 min cold (baracuda nvcc);
metal is unbuildable here — so per-step, verify the CPU path (fuel-ir+cpu-backend+fuel-core, ~40s)
and batch one cuda build to confirm the cross-backend re-points. This doc is the resume artifact —
it captures the code-grounded investigation (a 4-agent sweep) so the remaining steps don't need
re-deriving.

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

### B0.3–B0.5 cycle-free plan (from the 2026-06-26 sequencing investigation)

The fuel-ir grab-bag has **two dependency cycles** that must be broken before a clean topological
split: `{dyn_backend ↔ quantized}` (contract SCC — `dyn_backend::as_quantized_kernels` names
`quantized`; `quantized` traits are built on `dyn_backend`) and `{dtype ↔ cpu_storage}` (the
WithDType weld — `WithDType` has `HostBuffer`/`HostBufferRef` method sigs from `cpu_storage`;
`cpu_storage` needs `WithDType`). `cpu/mod.rs` + `cpu/kernels.rs` (VecOps SIMD) are pure leaves.
Layering apex→base: `storage` → `inplace_op` → `{dyn_backend↔quantized}` + `backend` → `{dtype↔cpu_storage}` → `cpu/`.

- **B0.4 — Break the WithDType weld (the safe, contained, UNBLOCKING next step; no crate moves).**
  Backends *consume* `WithDType`, they don't impl it, so this is small: split `WithDType`
  (dtype.rs:167-193) into a **pure-vocab core** (keeps `DTYPE`/`from_f64`/`to_f64`/`to_scalar`)
  + a new **`HostDType: WithDType`** extension carrying the 5 host-buffer methods (`cpu_storage_ref`
  / `to_cpu_storage_owned` / `to_cpu_storage` / `cpu_storage_as_slice` / `cpu_storage_data`,
  dtype.rs:184-192) — define `HostDType` in the `HostBuffer`-owning module (cpu_storage.rs for now).
  DROP the `+ crate::cpu::kernels::VecOps` supertrait (dtype.rs:177); relocate the obligation by
  adding an explicit `T: VecOps` bound at the ~5 fuel-cpu-backend sites that call `T::vec_dot`/
  `T::vec_reduce_*` (ops.rs:223/1041/1270/1353, conv2d.rs:348). Split the 2 `impl` macros
  (`with_dtype!` dtype.rs:197 ×11 types; `dummy_with_dtype!` dummy_dtype.rs:31 ×4) into vocab-half
  (stays) + host-half (moves with HostBuffer). Add `+ HostDType` at the 2 GPU call sites that use
  the host methods (fuel-cuda-backend/src/device.rs:708, **fuel-metal-backend/src/storage.rs:2035**).
  Result: `dtype` becomes a clean vocab leaf. **CAVEAT: the metal site can't be built on Windows —
  make the edit mechanically (precise site above) and verify cpu + cuda; metal verify needs a mac.**

- **B0.3 — Cut the `fuel-backend-contract` crate (above fuel-ir, below the backends).** Move the
  **11 contract-traits together** (they cross-reference, can't be split per-file): `DynBackendStorage`,
  `DynBackendDevice`, `BackendStorage`, `BackendRuntime`, `HostStorage`, `BackendCapabilityProvider`,
  `DynQuantizedStorage`, `QuantizedDeviceKernels`, `InplaceOp1/2/3`. The `{dyn_backend↔quantized}`
  cycle becomes *internal* to the new crate (fine). **KEEP these 5 data-vocabulary types in fuel-ir**
  (split them out of backend.rs into a vocab module): `FitStatus`, `BackendCapabilities`,
  `SubstrateClass`, `TransferPath`, `GgmlDType` — this resolves the error.rs `TransferPath` leak for
  free (it stays an intra-fuel-ir ref). Dep direction: backend-contract → fuel-ir; backends →
  backend-contract; **fuel-dispatch → backend-contract too** (it calls `BackendRuntime::would_fit`
  on `&dyn BackendRuntime` in chained_selector.rs:129 / vram_pressure_selector.rs:153). Implementers
  to re-point: every backend's dyn_impl/byte_storage. **COUPLING: `storage.rs` holds
  `Box<dyn DynBackendStorage>`, so moving the traits out of fuel-ir FORCES `storage.rs` to leave too
  (else `fuel-ir storage → contract → fuel-ir` cycle) — do B0.3 + the storage half of B0.5 together,
  or break storage→dyn_backend first.** Touches metal (unverifiable here).

- **B0.5 — Relocate storage-impl → fuel-memory.** `storage.rs` is a clean thin wrapper (only
  `dyn_backend` + `inplace_op` + op/conv/scalar/HostBuffer vocab; no dispatch/quantized/backend) —
  retarget `crate::dyn_backend`→`fuel_backend_contract`, op/conv/scalar stay at fuel-ir; merge with
  fuel-memory's byte-`Storage` (the Storage-unification; fuel-graph already targets
  `Arc<RwLock<fuel_memory::Storage>>`). Then `cpu_storage` (HostBuffer) + `cpu/` (VecOps SIMD) +
  `HostDType` + the host-half macro impls → fuel-memory (feasible once B0.4 broke the weld).

**Net partition (16 audited types): 11 traits → backend-contract; 5 data → stay fuel-ir.**
**A tiny cleanup to fold in: factories.rs's now-dead `BackendFactory::enumerate_devices`** (superseded
by fuel-hardware's HardwareEnumerator in B0.2a — harmless dead code).

## Verification crates (never workspace-wide — `tensor-tools` has a standing break)
`fuel-ir` (+ `--doc`), `fuel-memory`, `fuel-graph`, `fuel-dispatch`, `fuel-core` (+ `--doc` for
the granular re-exports & macro bodies in op.rs/tensor.rs), `fuel-cpu-backend`, `fuel-cuda-backend`
(heaviest macro user — live CUDA per environment-discipline DevShell incantations),
`fuel-vulkan-backend`, `fuel-inference`, `fuel-reference-backend --tests`.

## Pre-existing items surfaced (out of B0 scope, flag/fix separately)
- 3 broken fuel-core doctests (realize_f32-on-Result, above).
- `build_errors{,2,3,4}.txt` — tracked junk in repo root.
- `fuel-dispatch/src/fused.rs:1-15` module doc still says `fuel-storage::fused` (stale comment).
