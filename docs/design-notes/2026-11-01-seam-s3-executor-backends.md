# Seam S3: executor (fuel-tensor, fuel-dispatch) and the backends

ROADMAP row 137, first seam. Method from `2026-11-01-seams-and-cleanup-program.md`; numbers from
`scripts/structure_report.py` (definitions in `2026-11-01-structure-measurement.md`) at `origin/main`
`4569a530b` (0.35.14, script) and `5602a2c53` (0.35.15, the greps named below). Written 2026-11-01. Nothing here has been changed
yet; section 6 is the proposed order of small PRs. Leads are marked as such where they are not
measured.

## 1. What crosses the seam (method step 1)

LEFT = `fuel-tensor` (tier 100) and `fuel-dispatch` (tier 80); RIGHT = the seven backend crates
(tiers 40 to 90). Measured: LEFT uses RIGHT at 874 sites over 551 distinct items in 84 files
(top: `CudaDevice` 91, `CudaStorageBytes` 50, `VulkanStorageBytes` 46, `VulkanBackend` 29,
`CpuStorageBytes` 26); RIGHT uses LEFT at 33 sites over 11 items (`fuel_dispatch::kernel::KernelCaps`,
`OpParams`, `KernelBindingTable`: the vendor backends at tier 90 sit above dispatch at tier 80, so
this is the permitted direction). Up-tier sites: 0. Co-change: 19 of 102 executor-side commits also
touch a backend (19%), so the two sides already change mostly independently.

The seam has THREE parts, and they are different kinds of coupling:

1. **Storage.** `fuel_memory::BackendStorage` is a closed enum, one cfg-gated variant per backend
   (`Cpu`, `Cuda`, `Vulkan`, `Metal`), wrapped by `Storage` (bytes plus dtype tag). The executor
   matches on it directly: `BackendStorage::` appears 184 times in `fuel-dispatch/src/pipelined.rs`
   alone and 139, 63 and 37 times (Cpu, Cuda, Vulkan) across `fuel-dispatch/src`, 19, 3, 3 and 1
   across `fuel-tensor/src`.
2. **Devices and handles.** Finding the device handle a kernel needs: `find_cuda_device_in_cache`
   and `find_vulkan_backend_in_cache` (`pipelined.rs`, 9 call sites) search the executor's
   storage cache for any storage on the target backend and take its handle;
   `seed_extra_device_handles`, `seed_placed_device_handles` and `device_seed_storage`
   (`pipelined_bridge.rs`) exist to put such a storage in the cache by pushing a dummy `Op::Const`
   anchor node into the user's graph.
3. **Identity of a backend and a place.** `BackendId` (kind of backend) and `DeviceLocation`
   (kind plus ordinal) are both in `fuel-ir`, and the conversion between them is written out by hand
   in many places (section 3).

## 2. Two abstractions for the same thing (the main finding)

`fuel-backend-contract` defines trait objects for exactly this seam: `DynBackendStorage`,
`DynBackendDevice`, `BackendRuntime`, `BackendStreams`. They are implemented: `DynBackendStorage` for
the CPU, CUDA and Metal storages, `DynBackendDevice` for all four devices, `BackendRuntime` for the
CUDA and Vulkan devices and (in `fuel-tensor`) a `DeviceRuntimeHandle` wrapper. The Vulkan device's own source
says it "does not return DynBackendStorage objects". And the hot path does not use them:
`DynBackendStorage` appears once in `fuel-dispatch/src`, zero times in `fuel-tensor/src`.

The storage design (`docs/storage-unification.md`) chose the OTHER shape: a closed enum plus a
`BackendStorageOps` trait with convenience methods on the enum, so callers ask `storage.len_bytes()`
or `storage.copy_from(src)` and do not match. `BackendStorageOps` does not exist in the code (no
hit in any `.rs` file). So the enum shipped without the methods that were meant to keep callers
from matching on it, and a second, trait-object abstraction sits beside it, implemented but unused
by the executor.

This is the root of the measured backend-name counts: the executor has nothing to call except the
variants. Either abstraction would remove most of them; having both removes none. Which one is
intended is an architecture decision (section 7).

## 3. Leaks found by reading (method step 3)

- **Copied mapping tables.** `DeviceLocation -> BackendId` is written four times with the same four
  arms (`fuel-dispatch/src/optimize.rs`, `pipelined.rs` inline, `plan.rs` as `backend_for_device`,
  `fuel-tensor/src/pipelined_bridge.rs`). `BackendId -> DeviceLocation { gpu_id: 0 }` is written
  four times (`dispatch.rs`, `plan.rs` `default_device_for`, `topology.rs`, `fuel-graph/src/opt.rs`),
  all with the same arms and the same `_ => DeviceLocation::Cpu` fallback. Both types live in
  `fuel-ir`, so one definition there serves every crate. The copies agree, which is why this is
  behaviour-preserving; the `gpu_id: 0` default is also the place that has to change when devices
  are named by identity (row 125).
- **Backend-specific control flow in generic code.** `pipelined_bridge.rs` has a Vulkan branch in
  the realize path (`matches!(target_loc, DeviceLocation::Vulkan { .. })`), a Vulkan special case in
  `device_for_location` that returns an error telling the caller to use a different entry point, and
  per-backend arms in `device_seed_storage`. `fuel-dispatch/src/jit_ingest.rs` carries 70
  `cfg(feature = ...)` lines (the JIT path is CUDA-only today; not read in detail).
- **Handle discovery by searching a cache.** A handle is found by scanning the storage cache for
  any storage whose backend matches (and, since the row 114 fix, whose device matches). That makes
  "a device exists" a property of "some tensor happens to live on it", which is why the seeding
  anchors exist.
- **Index as identity.** `gpu_id` is a within-process ordinal and its order differs between builds
  on the development box (row 125). The `gpu_id: 0` defaults above assume otherwise.
- **Public internals.** `VulkanBackend` exposes its physical device as a public field; the Vulkan
  backend's `lib.rs` is about 14,000 lines.

## 4. What each side needs from the other (method step 4)

The executor needs, per device: allocate, upload, download, copy to another device, report free and
total memory, and a way to run a kernel it already holds a `KernelRef` for. It does not need to know
which backend it is talking to for any of those. The backends need, from the executor: the kernel
call signature (`KernelRef`, `OpParams`, layouts), capabilities and cost reporting. That is already
in `fuel-backend-contract` and `fuel-dispatch::kernel`; the second half of the seam is in order.
The first half (storage and device handles) is the part with no single interface in use.

## 5. The gap (method step 5)

| need | designed | exists | used by the executor |
| --- | --- | --- | --- |
| storage queries and copies | `BackendStorageOps` methods on the enum | no | no (matches) |
| per-device handle | `DynBackendDevice` | yes, 4 devices | 2 lines in `fuel-dispatch`, 16 in `fuel-tensor` mention it |
| storage as an object | `DynBackendStorage` | yes, 3 backends | 1 use |
| a registry of live devices | none designed | no | cache search plus anchor nodes |
| kind-of-backend to place and back | none | 8 copies | 8 copies |

## 6. Proposed order of PRs (method step 6)

Each is small, behaviour-preserving unless stated, red-first where behaviour could change, and
re-measured with `structure_report.py` (cfg lines and backend-path uses for the touched crate, and
the S3 item count).

1. **One mapping in `fuel-ir`:** `DeviceLocation::backend()` and `BackendId::default_location()`
   (gpu 0, CPU for a backend it does not know), replacing the eight copies. Round-trip and
   every-variant tests. Smallest and safest; also the one place the `gpu_id: 0` default will later
   change.
2. **Give `BackendStorage` the designed methods** (`len_bytes`, `location`, and the copy
   operation) and replace the `pipelined.rs` matches that only ask a question, not the ones that
   dispatch a kernel. Measure the drop in `BackendStorage::` uses.
3. **A device registry for one realize,** keyed by `DeviceLocation`, handed to the executor, in place
   of cache search, `seed_extra_device_handles`, `seed_placed_device_handles` and the anchor nodes.
   This changes behaviour at the seeding boundary and needs the row 114 two-GPU probe as its
   oracle (it failed on exactly this code before the fix).
4. **Identity in place of index** for the places that choose a device (row 125's selector).
5. **Park the trait-object alternative** (section 7): leave `DynBackendStorage` and the Vulkan
   exception as they are and decide deletion later, with reachability output.
6. **Move backend-gated code out of `fuel-dispatch`** (the JIT ingest path first) behind the
   backend crates' trait implementations.

Steps 1 and 2 need no decision; the storage decision (section 7) is made.

## 7. Decision (2026-11-01, CireSnave, relayed by the PM)

The question was whether the storage abstraction is (a) the closed enum plus ops methods, or (b)
trait objects. **Ruled: (a).** Keep the closed `BackendStorage` enum and write the
`BackendStorageOps` methods the design specifies; revisit trait objects only if backends are ever
to live outside Fuel. Row 138 may proceed on that basis.

What follows for this seam:

- Step 2 (the designed methods on `BackendStorage`) is now the main storage cleanup and needs no
  further decision. It is how the 184-site match surface shrinks.
- Step 5 changes: `DynBackendStorage` and the Vulkan exception to it are not adopted. They stay as
  they are and are recorded as the parked alternative; whether they are deleted is a separate
  decision, to be made with CodeRipper's reachability output in hand (it already reports several of
  these traits as having no callers).
- The seam contract for row 138 is therefore the enum plus the ops methods plus the kernel call
  signature, which means the backend crates stay in the Fuel workspace. A split of the backends
  into separate projects is off the table under this ruling; the kernel-runner seams that already
  exist (the `fuel-kernel-seam` crates) are unaffected.
- The leak list in section 3 still stands. The device registry (step 3) and identity in place of
  index (step 4) are independent of this ruling.

## 8. Limits

Read-through of the files named, not a full audit; `jit_ingest.rs`, `plan.rs` beyond the mapping
tables, and `scheduling.rs` were counted but not read; counts include `#[cfg(test)]` code in `src`;
the 184-site figure counts any line containing `BackendStorage::` (patterns, constructors and
returns alike) without classifying them.
