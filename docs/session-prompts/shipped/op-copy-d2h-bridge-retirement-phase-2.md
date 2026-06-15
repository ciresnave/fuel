# Session prompt — Op::Copy D2H (bridge-retirement Phase 2)

## What this session is for

Replace the per-backend D2H placeholder code shipped in commit `7a95001a`
with a graph-level `Op::Copy { target: DeviceLocation::Cpu }` insertion
at realize roots, dispatched through the binding table. This is **Phase 2
of the bridge-retirement trajectory** documented in
[ROADMAP.md §"Bridge-retirement trajectory post-9c"](../../ROADMAP.md#bridge-retirement-trajectory-post-9c).

Concretely, this session **deletes the per-variant `match self`** in
[fuel-storage/src/lib.rs::BackendStorage::read_to_cpu_bytes](../../fuel-storage/src/lib.rs#L146)
— including the Vulkan branch I wrote in `7a95001a`. After this lands,
device-to-host transfer is no longer a `BackendStorage` method; it's a
graph node whose kernel the optimizer plans and the executor dispatches
through the binding table, the same way every other op works.

This is the first session that actually deletes the bridge code from
`7a95001a`. The bridge was deliberately built as a placeholder; you're
ripping out the placeholder before more callers ossify around its shape.

## Why this matters architecturally

[01-identity](../architecture/01-identity.md) commits to "decisions
visible at the DAG-level optimizer." Today, D2H is a hidden post-realize
fixup invisible to the optimizer. After this session, D2H is an
`Op::Copy` node — the optimizer can see it, can reason about cost (a
4 GB tensor's D2H is the most expensive op in a realize call), can
fuse it with adjacent ops (e.g., D2H + dtype-cast on the host side), and
can route a single D2H to multiple consumers via Op::Copy fan-out.

This is also the **prerequisite for the dispatch-erased `Device` tag**
([05-backend-contract §"Static capability advertisement"](../architecture/05-backend-contract.md)
— Phase 5 of the bridge-retirement trajectory). Today `Device` carries
an `Arc<VulkanBackend>` because something downstream needs to call
`backend.download_bytes(s)`. Once D2H goes through the binding-table,
no caller reaches for the backend handle, and `Device` can shrink to
`{ backend_id, location }`.

## Read first (in this order)

1. **`ROADMAP.md` §"Bridge-retirement trajectory post-9c"** — the
   7-step path; this session is step 1 of that list.
2. **`docs/architecture/03-ir.md`** §"How nodes carry their op identity"
   — `Op::Copy` is already a primitive variant; the work is plumbing
   its kernel through the binding table.
3. **`docs/architecture/05-backend-contract.md`** §"Per-backend kernel
   registration" — the canonical-shape contract every backend's D2H
   kernel registration must satisfy. CPU is the source of truth.
4. **Commit `7a95001a`** (`feat(vulkan): VulkanBackendDevice + runtime
   Device wiring`) — the bridge code you're deleting. The Vulkan branch
   of `read_to_cpu_bytes` + the `upload_host_buffer` branches are the
   shape of "what the placeholder did." Step 1 of this session keeps
   `upload_host_buffer` alive (Phase 3 deletes it); step 1 only deletes
   the `read_to_cpu_bytes` branches.
5. **`fuel-storage/src/lib.rs`** lines 129–170 — `BackendStorage::
   read_to_cpu_bytes`. The method you're deleting.
6. **`fuel-core/src/pipelined_bridge.rs`** lines 93–130 —
   `realize_one_as_with_initial` and `realize_many_as_with_initial`.
   These currently call `read_to_cpu_bytes` directly at lines 105 and
   126. After this session, they emit an `Op::Copy { target: Cpu }` at
   the realize root and rely on the executor to land the CPU-side bytes
   in `storage_map`.
7. **`fuel-graph/src/lib.rs`** lines 653–675 — `Op::Copy { target:
   DeviceLocation }` + `Op::Move { target }`. The IR primitive already
   exists; no schema change there.
8. **`fuel-graph-executor/src/lib.rs`** lines 1859–1877 — the legacy
   executor's `Op::Copy` arm, which calls `backend.copy_to(&a.storage,
   &layout, *target)`. This is the **semantics you're porting** to the
   pipelined path; same input/output shape, same dtype, only residency
   changes.
9. **`fuel-storage/src/pipelined.rs`** §`compile_one` (line ~476) —
   the dispatcher of `WorkItemKind`. `Op::Copy` needs a new
   `WorkItemKind::Copy { target }` arm, or it can go through
   `WorkItemKind::Kernel` if you give it an `OpKind::Copy` and register
   D2H kernels per-backend. **Architecture-aligned: the second shape.**
   Justify the choice early in the session.
10. **`fuel-core-types/src/dispatch.rs`** lines 45+ — `enum OpKind`.
    You'll add `OpKind::Copy` here.
11. **`fuel-storage/src/dispatch.rs`** §`register_cpu_kernels`
    (line 2951) — pattern for adding the CPU D2H kernel registration
    (CPU→CPU is a memcpy noop; still register it for uniformity).
12. **`fuel-storage/src/vulkan_dispatch.rs`** + the audit memory entry
    [project_phase_7_6_step_9c_parity_audit.md](../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_phase_7_6_step_9c_parity_audit.md)
    "Vulkan binding-table audit shipped" follow-up — the canonical-key
    discipline the binding-table audit just established. Same protocol
    applies to your new D2H rows.

## What this session must NOT do

- **Don't delete `upload_host_buffer` or `alloc_zeroed_on`** in
  `fuel-core/src/pipelined_bridge.rs` / `fuel-core/src/inference_context.rs`.
  Those are Phase 3 of the bridge-retirement (H2D + zero-alloc through
  `Op::Alloc` + `Op::Copy`). They stay this session; only D2H moves.
- **Don't touch `Op::Move`**. Same kernel shape semantically, but
  `Op::Move` is a downstream rule-emitted op tied to the residency-
  eviction pass. Wire `Op::Copy` first; the executor's existing arm
  for `Op::Move | Op::Copy` (line 1862) is the precedent, but you're
  only migrating Copy this session.
- **Don't add a new `realize_as_keep_on_device` API**. The user-facing
  promise of `realize_one_as<T>` is "give me `Vec<T>` on the host."
  D2H stays implicit; you're just changing *how* the host bytes
  arrive (graph-emitted `Op::Copy` vs. ad-hoc `read_to_cpu_bytes`).
- **Don't register a CUDA-Vulkan or Vulkan-CUDA direct copy kernel**.
  Cross-GPU D2D is multi-GPU work (Phase 6c) and parked. Today the
  registered Copy kernels are device→CPU only.
- **Don't push to remote unless asked.**

## Concrete work — sequenced

### Step 1 — Decide: `WorkItemKind::Copy` vs. `OpKind::Copy` (~15 min, no code)

Two paths through the pipelined executor:

**A. Special-case `WorkItemKind::Copy { target }`.** `compile_one`
detects `Op::Copy` and emits a dedicated work-item kind, mirroring
`WorkItemKind::ReleaseMarker` from Phase B. `execute_work_item`
matches on it directly. **Pros**: simpler initial wiring. **Cons**:
backends register their D2H kernels through ad-hoc function pointers,
not through the binding table; the optimizer's per-decision-point
alternative resolution doesn't see `Op::Copy` as an op with rankable
alternatives.

**B. `OpKind::Copy` registered in the binding table.** Add `OpKind::
Copy` to `fuel-core-types::dispatch::OpKind`. Each backend registers
a `copy_to_cpu` kernel against the binding table with a canonical
key (likely `[dtype, dtype]` — input and output share dtype; residency
is encoded by the source storage's variant + the target on the op's
params). `compile_one` routes `Op::Copy` through the standard
`WorkItemKind::Kernel` arm. **Pros**: architecture-aligned. The
optimizer sees Copy as a peer of every other op; cost annotations
(Phase 7.6 step 8) attach to it; eventually copy/cast fusion rules
can see it. **Cons**: needs the binding-table key-shape canonicalized
(input residency is implicit in the source storage's BackendStorage
variant; target is in `OpParams` not the dtypes vector).

**Recommendation: Path B**, but write the chosen path's rationale into
the commit message and the memory entry. Path A is "do something
working fast"; Path B is "do something that fits the architecture."
The user's standing feedback (`feedback_architectural_cleanness_over_local_pragmatism`)
favors B. **Engage critically** — if Path A turns out to have a clean
upgrade path to B, that's a defensible incremental shape. If you
pick A, document the upgrade path explicitly.

### Step 2 — `OpKind::Copy` + binding-table key + CPU kernel (Path B path)

If Path A: skip to Step 3.

- Add `OpKind::Copy` to `fuel-core-types/src/dispatch.rs::OpKind` (the
  `#[non_exhaustive]` attribute means persisted profiles forward-parse).
  `as_str` arm: `"Copy"`.
- Add `Op::Copy { .. } => OpKind::Copy` to `op_to_op_kind` in
  `fuel-storage/src/pipelined.rs`.
- **Canonical CPU registration**: `table.register(Copy, &[dt, dt], cpu,
  copy_to_cpu_wrapper)` for each supported dtype. CPU→CPU is a
  memcpy through `Arc<[u8]>`; the wrapper handles every dtype the
  same way (it's bytes). One wrapper, registered across the
  bit-stable dtype set (F32, F64, BF16, F16, U32, U8 minimum;
  consider I32, I64, I8 since byte-storage handles them).
- Decision: should `Op::Copy`'s `OpParams` carry the target
  `DeviceLocation`? Read `Op::Copy { target }` — yes, it already
  does (the variant fields). The binding-table key (dtype, dtype)
  doesn't encode target; the dispatch wrapper reads it from
  `OpParams` to know what to allocate. This is the same shape
  `OpParams::QMatMul { quant_type }` uses.
- **Test**: round-trip a small F32 storage through `Op::Copy { target:
  Cpu }` in a fuel-storage unit test. Source storage on CPU; assert
  the result is a fresh `BackendStorage::Cpu` with identical bytes
  and that the source's Arc strong_count drops (the new node's
  Arc replaces the source in `storage_map`).

### Step 3 — Per-backend D2H kernels

For each backend with a D2H path, register a Copy kernel that produces
a CPU storage from a same-backend source. The binding-table key shape
matches CPU's `[dt, dt]`; the dispatch wrapper does the actual download.

- **CUDA**: in `fuel-storage/src/dispatch.rs::register_cuda_kernels`
  (the `#[cfg(feature = "cuda")]` block near line 5091), add Copy
  registrations. The wrapper extracts the `CudaStorageBytes` from the
  input, calls `s.to_cpu_bytes()` (same one `read_to_cpu_bytes` used
  for the CUDA branch), and constructs a `BackendStorage::Cpu` from
  the result. Same dtype set as CPU.
- **Vulkan**: in `fuel-storage/src/vulkan_dispatch.rs::register_vulkan_kernels`,
  add Copy registrations. The wrapper extracts the `VulkanStorageBytes`,
  reaches its attached `Arc<VulkanBackend>` handle (the `s.backend()`
  accessor — see `read_to_cpu_bytes`'s current Vulkan branch), calls
  `backend.download_bytes(s)`, constructs the CPU storage. Same dtype
  set.
- **Metal**: stub. `read_to_cpu_bytes`'s current Metal branch errors
  with "Metal A4 D2H substrate not yet wired"; the Copy kernel
  registration should mirror that — return the same error. When Metal
  D2H ships, the kernel becomes real with zero call-site change.

**Binding-table audit discipline** (per the new
[`project_phase_7_6_step_9c_parity_audit.md`](../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_phase_7_6_step_9c_parity_audit.md)
Vulkan-audit-shipped follow-up): every backend's registration must
use the same `[dt, dt]` shape as CPU. The wrapper differs per backend
but the binding-table key is identical. Add a parity-check unit test
that asserts `lookup_alternatives(OpKind::Copy, &[F32, F32], Vulkan)`
returns 1 alternative when built `--features vulkan` (mirroring the
existing `vulkan_dispatch_softmax_norm_registered` pattern).

### Step 4 — `realize_one_as` emits `Op::Copy` instead of calling `read_to_cpu_bytes`

In `fuel-core/src/pipelined_bridge.rs::realize_one_as_with_initial`
and `realize_many_as_with_initial`:

Today:

```rust
let (storage, _layout) = PipelinedExecutor::realize(graph.clone(), target, cache)?;
let guard = storage.read().map_err(...)?;
let bytes = guard.inner.read_to_cpu_bytes()?;
Ok(bytemuck::cast_slice::<u8, T>(&bytes).to_vec())
```

After:

```rust
// If the realize target's device differs from CPU, splice an
// Op::Copy { target: Cpu } between the target and the realize call.
let cpu_target = if backend_id != BackendId::Cpu {
    let mut g = graph.write()...;
    g.push(Node { op: Op::Copy { target: DeviceLocation::Cpu },
                  inputs: vec![target],
                  shape: g.node(target).shape.clone(),
                  dtype: g.node(target).dtype, ... })
} else {
    target
};
let (storage, _layout) = PipelinedExecutor::realize(graph.clone(), cpu_target, cache)?;
let guard = storage.read()...;
// `storage` is now BackendStorage::Cpu — extract bytes directly
let bytes = match &guard.inner {
    BackendStorage::Cpu(s) => s.bytes(),
    other => return Err(...) // unreachable: post-Copy must be CPU
};
Ok(bytemuck::cast_slice::<u8, T>(bytes).to_vec())
```

(Use `realize_many` for the multi-target variant; splice one Op::Copy
per target whose backend != CPU.)

Mirror this for `realize_many_as_with_initial`. **Don't graph-mutate
twice** — fold the Op::Copy splice into the same write-lock segment
where `ensure_target_backends` already takes one (`prepare()` in
`pipelined_bridge.rs`).

**Watch for**: the Op::Copy node needs its own `target_backend` set so
the executor knows which backend's Copy kernel to dispatch. The
source-backend's Copy kernel is what runs (it downloads from its own
storage); set `target_backend` on the Op::Copy node to match the
*source* node's `target_backend` (the source dictates which backend
owns the kernel — Vulkan-resident input → Vulkan's `copy_to_cpu`
kernel runs). `prepare()`'s `ensure_target_backends` walk needs to
include the spliced Copy in its iteration.

### Step 5 — Delete `BackendStorage::read_to_cpu_bytes`

Once steps 1-4 are green:

- Delete the method body in `fuel-storage/src/lib.rs` (lines 129-170).
- Delete the `read_to_cpu_bytes_cpu_variant` test (lines 268-280) —
  it asserts an internal detail that no longer exists. Replace it
  with a fuel-storage-level test that round-trips a CPU Storage
  through `Op::Copy { target: Cpu }` and asserts byte equality.
- Search the workspace for other `read_to_cpu_bytes` call sites
  with grep. The only remaining one should be the new Copy kernel
  wrappers themselves (which call backend-internal `to_cpu_bytes()` /
  `download_bytes()`, *not* the trait method you're deleting).
  Anything that still calls `BackendStorage::read_to_cpu_bytes`
  outside the kernel wrappers needs its own migration to Op::Copy.

This is the **first deletion of bridge code from `7a95001a`** the
session was contracted to deliver.

## Test budget

After each step:

- `cargo test -p fuel-storage --lib` — schema + binding-table tests.
- `cargo test -p fuel-storage --test vulkan_dispatch_live` (and
  `-- --ignored` for the RTX 4070 live ones) — Vulkan Copy kernel
  registration + execution.
- `cargo test -p fuel-storage --features cuda --lib` — CUDA Copy
  kernel registration. If you have CUDA built, the dispatch tests
  pick it up.
- `cargo test -p fuel-core --features vulkan --lib` —
  `pipelined_bridge` integration through the user-facing
  `Tensor::realize_*` surface.
- `cargo test -p fuel-core --features vulkan --test flash_attn_vulkan`
  — end-to-end Vulkan-resident model realize.
- `cargo test -p fuel-core --features vulkan --lib forward_with_kv_context_vulkan_matches_cpu`
  — the KvCache + Vulkan parity test from `7a95001a`. **Must stay
  green throughout** — this is the contract test for the post-9c
  Vulkan path.

Final sweep (all defaults on the dev rig):

- `cargo test --workspace --features cuda,vulkan` (don't run with
  `--all-features`; some feature combos in the workspace are
  incompatible — see CI configs).

## Success criteria

- `OpKind::Copy` exists (if Path B chosen) and is reachable from
  `op_to_op_kind`.
- Each backend (CPU, CUDA, Vulkan; Metal-stub) registers a Copy
  kernel at the canonical `[dt, dt]` key for every dtype the
  byte-storage substrate supports.
- `realize_one_as<T>` and `realize_many_as<T>` splice
  `Op::Copy { target: Cpu }` at non-CPU realize roots; the executor
  produces a `BackendStorage::Cpu` for the spliced node; D2H is no
  longer a post-realize fixup.
- `BackendStorage::read_to_cpu_bytes` is **gone** from
  `fuel-storage/src/lib.rs`. The Vulkan branch I wrote in
  `7a95001a` is gone with it.
- All test sweeps above are green; `forward_with_kv_context_vulkan_matches_cpu`
  in particular still passes within `5e-3 abs / 1e-2 rel`.
- ROADMAP `Current frontier` updated to reflect Phase 2 of bridge-
  retirement shipped; the bridge-retirement trajectory subsection
  updated to mark step 1 ✅ shipped and to point at step 2 (H2D
  + zero-alloc through Op::Alloc) as next.
- Memory entry [project_phase_7_6_step_9c_parity_audit.md](../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_phase_7_6_step_9c_parity_audit.md)
  updated with a "Bridge-retirement Phase 2 (Op::Copy D2H) shipped"
  follow-up. Cover: which OpKind was added, which backends registered
  Copy kernels, what got deleted from `7a95001a`, what test surface
  exercises the new path.
- Short audit-style note in the commit message: what changed, what
  was deleted, link to the architecture-doc check this work advanced
  (identity check #1, more decisions visible to the DAG-level
  optimizer; backend-contract check, every primitive op including
  Copy now has a CPU bit-stable kernel).

## Architecture-alignment check

Per [01-identity §"How this identity is enforced"](../architecture/01-identity.md#how-this-identity-is-enforced):

- **Check #1 — More decisions visible to the DAG-level optimizer**:
  ✅ Op::Copy is now a graph node, not a post-realize fixup. The
  optimizer can see it, cost it, fuse it.
- **Check #2 — Fewer dispatch-time branches**: ✅ `read_to_cpu_bytes`'s
  per-backend `match self` is gone. D2H dispatches through the
  binding table like any other op.
- **Check #4 — Backend-contract enforceable as a CI lint**: ✅
  the new Copy kernels participate in the existing `register_cpu_kernels`
  coverage lint (line ~5937 of dispatch.rs lists the OpKinds every
  primitive must have a CPU implementation for — add `OpKind::Copy`
  to that list).

## When this session ships, the next gate

Phase 3 of the bridge-retirement trajectory: **H2D + zero-alloc through
`Op::Alloc` + `Op::Copy`**. Same shape as this session — add `Op::Alloc
{ shape, dtype, device }` and `Op::ZeroFill` as primitives; each backend
registers `alloc` + `zero_fill` kernels; callers (KvCache::with_capacity,
build_const_cache_from_graph) emit these instead of calling
`alloc_zeroed_on` and `upload_host_buffer` directly. **Deletes**:
`alloc_zeroed_on` (inference_context.rs:303) + `upload_host_buffer`
(pipelined_bridge.rs:258) including the Vulkan + CUDA branches in each.
That's the second deletion of bridge code from `7a95001a`.
