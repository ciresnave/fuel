# Session prompt — SystemTopology service

## What this session is for

Build a first-class **`SystemTopology`** service that is the single
source of truth for: which devices exist on this host, which
backends are loaded into this process, which backends can target
which devices, which backends share Storage substrate (so
cross-backend on the same device is free), and what transfer paths
exist between devices.

This is a **focused, narrow session.** Build the service, expose
its query API, test that every predicate returns sane answers on
the dev box, ship. Do NOT wire it into the picker, the Op::Copy
planner, the optimizer, or anything else — those consumers come in
their own sessions. The discipline matters: building consumers
alongside the service would design SystemTopology around one
consumer's needs and bake in coupling we'd regret later.

## Why this session exists

The 2026-05-30 picker-alternatives audit (see
[`judge-alternatives-picking-audit-results.md`](
./judge-alternatives-picking-audit-results.md))
identified that multiple in-flight pieces need the same body of
knowledge and there's no clean home for it:

1. **The picker** (Phase 7.6 step 9b's `resolve_kernel` and its
   future enhancements) needs to enumerate alternatives across all
   `BackendId`s that share a target device — so AOCL/MKL/portable-CPU
   can compete on the same CPU work without each being a separate
   monolithic realize.
2. **The Op::Copy planner** (graph-level transfer insertion) needs
   to know whether a cross-backend edge requires a real copy (cross
   `DeviceLocation`), is free (same `DeviceLocation` with shared
   Storage substrate — the CPU trio), or requires special handling
   (same physical hardware reached via different drivers, e.g.
   Vulkan vs CUDA on the same GPU).
3. **The cost model** (Layer-1 static costs in `fuel-storage::cost`)
   needs transfer-cost estimates to factor cross-device decisions
   into the static estimate.
4. **Diagnostics** — error messages like "no backend for
   `(MatMul, [F32×3])` on this host" should be able to enumerate
   what backends DID register and what coverage they have.

Today this knowledge is fragmented across at least five places:

- `BackendCapabilities` in `fuel-core-types::backend` — per-backend
  advertisement.
- `TransferPath` enum in `fuel-core-types::backend` — the path
  taxonomy already exists.
- `CapabilityRegistry` in `fuel-storage::dispatch` — global
  per-backend registry.
- `Router::capability_index` in `fuel-graph-router` — capability ↔
  device map, built piecemeal by `add_cpu()` / `add_cuda()` etc.
- `ProbeReport` in `fuel-core::probe` — device enumeration with
  equivalence classes.
- `KernelBindingTable` in `fuel-storage::kernel` — implicit "what
  kernels are registered for which `BackendId`."

Each component knows part of the picture; none of them composes
the whole. `Router`'s `capability_index` comes closest, but it's
coupled to `Router : GraphBackend` (on the retirement path per
Phase 7.6 step 9c) and only knows what backends were explicitly
`add_*`-ed at Router construction — not what's actually loaded in
the process.

## Background — what already exists

Read these in order:

- [`fuel-core-types/src/backend.rs`](
  ../../fuel-core-types/src/backend.rs) — `BackendCapabilities`,
  `BackendCapabilityProvider`, `TransferPath`. The advertisement
  shape; this stays as the per-backend declaration. SystemTopology
  collects these.
- [`fuel-core-types/src/probe.rs`](
  ../../fuel-core-types/src/probe.rs) — `BackendId`, `DeviceDescriptor`,
  `EquivalenceKey`. Device-enumeration types; SystemTopology
  composes ProbeReport's output.
- [`fuel-core/src/probe.rs`](../../fuel-core/src/probe.rs) — the
  `ProbeReport::probe_all()` collector that already walks every
  loaded backend's `enumerate_devices()`. SystemTopology should
  call this (or share its mechanism) for device discovery.
- [`fuel-storage/src/dispatch.rs`](../../fuel-storage/src/dispatch.rs) —
  `CapabilityRegistry` (lines ~39-95), `global_registry()`,
  `register_backend_capabilities()`, the process-wide capability
  store. SystemTopology should read from this for the backend
  side.
- [`fuel-graph-router/src/lib.rs`](../../fuel-graph-router/src/lib.rs)
  lines 637-786 — `Router::capability_index`, `devices_for`,
  `supports`, `register_capabilities`. This is the closest thing
  to a topology service today; understand what it does and
  doesn't cover.
- Memory entries: `project_phase6b_probe_judge_dispatch`,
  `project_phase_7_6_step_4_in_progress`,
  `project_judge_alternatives_audit` (this session's reason for
  existing).

## What SystemTopology must provide

The query API is the contract. Sketch:

```rust
pub struct SystemTopology { /* private */ }

impl SystemTopology {
    /// Build the topology from probe data + the global capability
    /// registry + the global binding table. Eager-init on first
    /// access via the same `OnceLock` pattern `global_bindings`
    /// uses; an explicit `refresh()` exists for the rare case
    /// (test setup, late backend registration) where re-scanning
    /// is needed.
    pub fn current() -> &'static SystemTopology;
    pub fn refresh();

    // --- enumeration ---

    /// Every `DeviceLocation` known to this process.
    pub fn devices(&self) -> &[DeviceLocation];

    /// Every `BackendId` whose kernels are registered.
    pub fn backends(&self) -> &[BackendId];

    // --- the load-bearing predicates ---

    /// Which backends can target this device? CPU might return
    /// `[Cpu, Aocl, Mkl]`; `Cuda { gpu_id: 0 }` returns `[Cuda]`
    /// today; `Vulkan { gpu_id: 0 }` returns `[Vulkan]`.
    pub fn backends_for(&self, dev: DeviceLocation) -> &[BackendId];

    /// Which devices can this backend target? `Cuda` returns
    /// `[Cuda { gpu_id: 0 }, Cuda { gpu_id: 1 }, ...]`.
    pub fn devices_for(&self, backend: BackendId) -> &[DeviceLocation];

    /// **Critical predicate.** Do these two backends operate on
    /// the same Storage substrate when targeting the same device?
    /// `shares_storage(Cpu, Aocl) == true` — the same
    /// `CpuStorageBytes` flows through both; calling an AOCL
    /// kernel after a portable-CPU kernel is a vtable swap, no
    /// data movement. `shares_storage(Cuda, Vulkan) == false`
    /// even when both target the same physical GPU — they have
    /// distinct allocators / pointer namespaces.
    pub fn shares_storage(&self, a: BackendId, b: BackendId) -> bool;

    /// What's needed to move bytes from src to dst?
    /// `transfer_path(Cpu, Cpu) == TransferPath::SameDevice`.
    /// `transfer_path(Cuda{0}, Cuda{1}) == TransferPath::Peer` if
    /// CUDA P2P is enabled on this host; else `HostStaging`.
    /// `transfer_path(Cuda{0}, Vulkan{0})` returns `HostStaging`
    /// today (external-memory import is out of scope, see below).
    pub fn transfer_path(
        &self, src: DeviceLocation, dst: DeviceLocation,
    ) -> TransferPath;

    // --- diagnostics ---

    /// Per-backend `BackendCapabilities` snapshot. The picker can
    /// read precision claims / op coverage per backend without
    /// re-querying the binding table.
    pub fn capabilities(&self, backend: BackendId)
        -> Option<&BackendCapabilities>;
}
```

The exact field names + return types are the session's call;
above is the shape. Document the rationale wherever a non-obvious
choice gets made.

## Architectural decisions to surface explicitly

Each is the session's to resolve, but flag the choice in the
commit message + memory entry so future sessions can find the
rationale.

### TDP-1: Where does SystemTopology live?

- **A) `fuel-core::topology`** (new module, fuel-core).
  Probably right — needs `ProbeReport` (in fuel-core), needs
  `global_bindings()` + `global_registry()` (in fuel-storage,
  re-exported through fuel-core). Same dependency height as
  `fuel-core::dispatch` (the Judge consumer).
- **B) Split: types in `fuel-core-types::topology`, builder in
  `fuel-core::topology`.** Cleaner if downstream callers in
  fuel-graph or fuel-storage need to type-mention `SystemTopology`
  without taking the full fuel-core dep — unlikely today but
  forecast.
- **C) Standalone `fuel-topology` crate.** Premature; only do this
  if option A causes a real dep cycle.

**Recommendation:** A. Reassess if the picker session pulls
SystemTopology into fuel-storage and needs the types separated.

### TDP-2: How is "shares Storage substrate" encoded?

The CPU trio (Cpu/Aocl/Mkl) share `CpuStorageBytes`. The future
Metal might share with itself only. CUDA shares with CUDA on the
same gpu_id. Vulkan shares with Vulkan on the same gpu_id.

- **A) Hardcoded table in SystemTopology::build().** Fast to write,
  fragile to extend. Every new backend requires editing the table.
- **B) New field on `BackendCapabilities`:**
  `storage_substrate: SubstrateClass`. Backends self-declare. The
  CPU trio all return `SubstrateClass::HostBytes`; CUDA returns
  `SubstrateClass::CudaUntyped`; Vulkan returns
  `SubstrateClass::VulkanBuffer`. `shares_storage(a, b) ==
  caps(a).storage_substrate == caps(b).storage_substrate &&
  same_device_location`. Forward-extensible.
- **C) Compute from the existing `BackendStorage` variant the
  backend produces.** Robust but requires introspection on a
  backend-specific type — likely not feasible from
  fuel-core-types.

**Recommendation:** B. The extra `SubstrateClass` enum is ~one
declaration per backend in their capability provider; the cost is
tiny and the predicate becomes trivially correct.

### TDP-3: Multi-device handling (CUDA gpu_id 0 vs 1)

Two CUDA GPUs: `shares_storage(Cuda, Cuda)` is ambiguous without
device info. The fix is the predicate takes `(BackendId,
DeviceLocation)` not just `BackendId`. Two storages on
`Cuda { gpu_id: 0 }` share; `Cuda { gpu_id: 0 }` vs
`Cuda { gpu_id: 1 }` don't — they need `TransferPath::Peer` or
`HostStaging`.

**Decision:** make the shares_storage signature
`shares_storage(a: (BackendId, DeviceLocation), b: (BackendId,
DeviceLocation)) -> bool`. The CPU trio returns true when both
devices are `DeviceLocation::Cpu` (always true today since CPU is
the singleton). Future NUMA-split CPU would distinguish.

### TDP-4: TransferPath enumeration vs cost estimates

`TransferPath` is already an enum with 5 variants. Bandwidth +
latency estimates were called out as Phase A5 work in
`fuel-core-types::backend` doc comments but haven't shipped.

- **A) Just enumerate paths.** SystemTopology returns the
  TransferPath enum value; the cost model later attaches numbers.
- **B) Bundle numeric cost.** Return
  `TransferEstimate { path, bandwidth_gbps, latency_us }`. Needs
  measurements; can fall back to hardcoded defaults per path
  variant.

**Recommendation:** A for this session. The picker doesn't need
numeric costs yet — the path discriminator alone tells it
"trivially-free vs cheap vs expensive." Add numeric refinement in
a later session when the cost model demands it.

### TDP-5: Lifecycle — topology must reflect device/backend changes

**This is load-bearing.** SystemTopology is NOT a snapshot. The
host's view of what backends are loaded and what devices are
available changes during a process's lifetime, and consumers
must see those changes without restart:

- A backend completes lazy registration via
  `extend_global_bindings` after the picker has already run once.
- A backend's loader (AOCL DLL, CUDA runtime, Vulkan ICD) succeeds
  on a deferred load attempt.
- A device fails / hangs / is hot-unplugged (rare but matters for
  long-running serving processes).
- A device is added (eGPU plug-in, rare).
- Tests inject or remove mock backends between test cases.

The right shape is **generation-counter invalidation, not
snapshot.** Sketch:

```rust
// Process-wide monotonic counter; every component that can
// change the topology bumps it.
static TOPOLOGY_GENERATION: AtomicU64 = AtomicU64::new(0);

static CURRENT_TOPOLOGY: RwLock<Option<Arc<SystemTopology>>>
    = RwLock::new(None);

pub fn bump_topology_generation() {
    TOPOLOGY_GENERATION.fetch_add(1, Ordering::Release);
}

impl SystemTopology {
    /// Returns a snapshot-Arc valid for the caller's use. If the
    /// generation counter has advanced since the cached topology
    /// was built, rebuilds and atomically swaps. Cheap when
    /// nothing's changed (one atomic load + Arc clone).
    pub fn current() -> Arc<SystemTopology> {
        let gen = TOPOLOGY_GENERATION.load(Ordering::Acquire);
        if let Some(t) = CURRENT_TOPOLOGY.read().unwrap().as_ref() {
            if t.generation == gen {
                return t.clone();
            }
        }
        let fresh = Arc::new(SystemTopology::build_at(gen));
        *CURRENT_TOPOLOGY.write().unwrap() = Some(fresh.clone());
        fresh
    }
}
```

The Arc-return is important: a long-running consumer (a picker
walking a large graph) gets a stable view for the duration of its
call even if a backend registers mid-walk. Next call sees the new
state. No torn reads.

**Who bumps the generation:**

- `register_backend_capabilities` (existing function in
  `fuel-storage::dispatch`) — bump on registration.
- `extend_global_bindings` — bump after the closure runs.
- A new `bump_topology_generation()` exposed for tests + future
  device-loss detection paths.
- The probe path on re-probe (a separate session will wire
  hot-unplug detection; for now `ProbeReport::probe_all` is
  called once at build).

This session ships the generation-counter mechanism and the bumps
at the existing registration sites. **Physical device-loss
detection (a GPU disappearing mid-run) is its own work and not
this session's scope** — it requires backend-level failure-
detection plumbing that doesn't exist yet. Surface the seam (a
public `bump_topology_generation()` plus the rebuild-on-stale
contract) so the loss-detection session has a clean place to hook
in.

### TDP-6: Atomicity + concurrent access

The rebuild needs to be safe against concurrent `current()` calls.
The `RwLock<Option<Arc<...>>>` pattern above gives:

- Multiple concurrent readers when topology is current: no
  contention beyond an `AtomicU64::load`.
- One writer when rebuilding; readers spin briefly. Rebuild is
  fast (probe + binding-table walk, sub-millisecond).
- The returned `Arc<SystemTopology>` lives independent of the
  RwLock — no risk of deadlock if a consumer holds the topology
  while calling a function that triggers a rebuild.

**One subtle hazard to handle:** during rebuild, the
`global_bindings()` and `global_registry()` reads should happen
*after* the generation snapshot, so we don't cache a topology
labelled with generation N but built from registry state at
generation N+1. The pattern:

```rust
fn build_at(_caller_gen: u64) -> SystemTopology {
    // Re-read the live counter inside the build to capture the
    // generation we actually built against. May differ from the
    // caller's snapshot if another bump landed mid-build; that's
    // OK — the next current() call will see the higher counter
    // and rebuild again.
    let built_gen = TOPOLOGY_GENERATION.load(Ordering::Acquire);
    // ... build from current global state ...
    SystemTopology { generation: built_gen, /* ... */ }
}
```

Document this in the implementation; a stale build is fine
(self-healing on next access), a mislabelled build is not.

### TDP-7: BackendCapabilities op_dtype_support vs KernelBindingTable

`BackendCapabilities::op_dtype_support` is a `HashSet<(OpKind,
DType)>` advertised by the backend. `KernelBindingTable` is what
actually got registered. These can diverge: a backend that
declares `(MatMul, F32)` support but forgot to register the
wrapper would have inconsistent state.

**Recommendation:** SystemTopology should treat the
KernelBindingTable as the source of truth for "what kernels exist"
and surface BackendCapabilities's other fields (alignment,
granularity, transfer paths) as the advertisement layer. A
diagnostic test in this session should assert
`op_dtype_support ⊆ binding_table_coverage` per backend — flag
divergence early.

## Scope of work

### Step 1 — type design

- Decide TDP-1 / TDP-2 (most consequential up front).
- If TDP-2 is B, add `SubstrateClass` enum to
  `fuel-core-types::backend` and add the field to
  `BackendCapabilities`. Touch every existing capability provider
  (CpuBackend, CudaBackend, VulkanBackend, AoclBackend,
  MklBackend) to declare it. **This is the one place this session
  legitimately touches sibling crates** — add the field, set the
  obvious value. Don't ask the backend to make a substrate
  policy decision; just label what's already true.
- Sketch `SystemTopology` struct + query method signatures.

### Step 2 — builder

- `SystemTopology::current()` → `OnceLock`-init.
- Builder pulls from:
  - `ProbeReport::probe_all()` for device enumeration (one call;
    the report is already cached internally).
  - `global_registry()` for `BackendCapabilities` per backend.
  - `global_bindings()` for op-coverage cross-check (TDP-7).
- Build the inverted indices: `backends_for_device`,
  `devices_for_backend`, the shares_storage adjacency.
- Build the `transfer_path` matrix from
  `BackendCapabilities::transfer_paths`.

### Step 3 — predicate implementations

- `backends_for(dev)`, `devices_for(backend)` — direct map
  lookups.
- `shares_storage(a, b)` — substrate match + device equality.
- `transfer_path(src, dst)` — matrix lookup; default to
  `TransferPath::HostStaging` if no specific entry (every backend
  supports host staging by contract).
- `capabilities(backend)` — direct lookup.

### Step 4 — tests

The dev box has CPU + CUDA + Vulkan (per `project_dev_environment`).
The session should write tests that, conditional on feature flags,
assert the topology reports the expected shape:

- **Always-on (CPU only):** `devices()` contains
  `DeviceLocation::Cpu`; `backends()` contains `BackendId::Cpu`;
  `backends_for(Cpu)` includes `Cpu`; `shares_storage((Cpu, Cpu),
  (Cpu, Cpu))` is true.
- **`cfg(feature = "cuda")`:** `backends()` contains `Cuda`;
  `backends_for(Cuda { gpu_id: 0 })` is `[Cuda]`;
  `transfer_path(Cuda { gpu_id: 0 }, Cpu) == HostStaging` (or
  `DeviceCopy` — whatever baracuda advertises).
- **`cfg(feature = "vulkan")`:** analogous to CUDA but for Vulkan.
- **`cfg(all(cuda, vulkan))`:** `transfer_path(Cuda{0}, Vulkan{0})
  == HostStaging` (per the audit, external-memory import is
  out-of-scope).
- **`cfg(feature = "aocl")` (or skip on Windows where AOCL DLL
  isn't on PATH per the memory):** `backends_for(Cpu)` includes
  both `Cpu` and `Aocl`; `shares_storage((Cpu, Cpu), (Aocl, Cpu))`
  is true.
- **Divergence guard (TDP-7):** for each registered backend, every
  `(op, dtype)` in `op_dtype_support` resolves in
  `KernelBindingTable`. Fail with a useful message if not.
- **Live-update (TDP-5):** call `SystemTopology::current()` to
  snapshot the topology; register a new mock backend via
  `extend_global_bindings` + `register_backend_capabilities`;
  call `current()` again and assert the new backend is visible
  AND that the second `Arc` is a distinct allocation from the
  first (proving the rebuild fired). A separate test asserts that
  two `current()` calls in a row with no intervening change
  return the same `Arc` (no spurious rebuilds).
- **Concurrent-access guard:** spawn N threads each calling
  `current()` in a loop; from another thread, periodically bump
  the generation counter. Assert no panics, every returned `Arc`
  is internally consistent (the predicates it answers match its
  reported `generation`), and the registered-backend count is
  monotonically non-decreasing across observed generations.

### Step 5 — memory + docs

- New memory entry `project_system_topology_shipped.md` capturing
  what landed + TDP resolutions + what's deliberately deferred.
- Update `project_judge_alternatives_audit.md`'s "recommended
  ordering" note: SystemTopology is now Session 0; A/B/C/D
  follow.
- A short doc comment on `SystemTopology::current()` pointing at
  this prompt for the rationale.

## What's NOT in scope

The discipline guard. The session must resist all of these
even when "it would only take a few minutes":

- **Wiring SystemTopology into the picker, planner, or
  optimizer.** Those are separate sessions. The picker session
  (audit's Session A) will be the first consumer; let it.
- **Op::Copy planner.** Topology provides the predicates the
  planner needs; building the planner itself is later work.
- **Retiring `Router::capability_index` / `Router::devices_for`.**
  Keep both for now. Router's path is on the 9c retirement
  trajectory anyway; let topology stand up first, prove out, then
  retire Router's piece in its own session.
- **External-memory import for cross-vendor GPU sharing
  (VK_KHR_external_memory + cudaImportExternalMemory).** Per the
  picker-audit discussion, treat Vulkan↔CUDA on the same physical
  GPU as `HostStaging`. Build the import path only when a real
  workload demands it.
- **Numeric transfer-cost estimates (TDP-4 option B).** Defer; the
  enum discriminator is enough for now.
- **NUMA-split CPU, multi-socket awareness.** Today
  `DeviceLocation::Cpu` is a singleton; that's fine. Numa support
  is a separate Phase 7b expansion.
- **Physical device-loss detection.** A GPU disappearing mid-run
  (driver crash, hot-unplug, ECC failure) requires backend-level
  failure-detection plumbing that doesn't exist today. This
  session ships the generation-counter mechanism + the
  `bump_topology_generation()` seam so the loss-detection session
  has a clean place to hook in; building the detector itself is
  later work.
- **Backend-failure recovery.** When a device disappears, who
  re-routes the in-flight graph? That's executor + planner
  policy, not topology's concern.
- **Documenting the picker's future consumption pattern.** The
  picker session will write that doc when it uses the API for
  real. Topology doesn't predict.

## Deliverables

1. **New module** (probably `fuel-core/src/topology.rs`) with the
   `SystemTopology` type + builder + query API.
2. **`SubstrateClass` enum** (if TDP-2 = B) added to
   `fuel-core-types::backend`, with the corresponding field on
   `BackendCapabilities` and updates to every backend's
   `BackendCapabilityProvider` impl. Single-line additions per
   backend.
3. **Tests** per Step 4, gated by feature flags so the CPU-only
   build still passes them all.
4. **Memory entry** `project_system_topology_shipped.md`.
5. **Update** to `project_judge_alternatives_audit.md` noting that
   SystemTopology is the new Session 0 prerequisite.

## Scope estimate

- Step 1 (type design): ~30 min.
- Step 2 (builder): ~60-90 min, mostly composing existing sources.
- Step 3 (predicates): ~30 min — they're direct lookups.
- Step 4 (tests): ~60 min, plus the per-feature-flag verification.
- Step 5 (memory + audit doc update): ~30 min.

**Total:** 1 focused session, 2-3 commits.

- Commit 1: `SubstrateClass` addition + backend capability provider
  updates (if TDP-2 = B).
- Commit 2: `SystemTopology` module + builder + predicates.
- Commit 3: Tests + memory + audit doc update (optional split from
  commit 2).

## Why this session, this scope, this order

The picker-audit's Session C ("graph-aware Router for the
binding-table world") originally assumed it would build the
topology knowledge as it went. That's wrong for the same reason
the audit itself was warranted: doing exploratory architectural
work alongside the consumer that needs it produces a topology API
shaped to that consumer's first usage pattern, which won't fit the
second consumer.

Building SystemTopology first, with deliberately *no consumer in
the same session*, forces an API designed against the predicate
contract rather than against the picker's first call site. The
picker, planner, cost model, and diagnostics then each consume the
same predicates without renegotiating the surface.

The temptation will be to "just sketch in the picker call" while
you're already in the topology code. **Don't.** That's exactly the
coupling this session-split is designed to prevent. The picker
sessions are next; they will use what's here.

## Pointers

- Audit driving this work: [`judge-alternatives-picking-audit-results.md`](
  ./judge-alternatives-picking-audit-results.md), specifically the
  "gaps in the endpoint architecture" + "graph topology awareness"
  discussion.
- Existing primitive types:
  `fuel-core-types/src/backend.rs` (BackendCapabilities,
  TransferPath, BackendCapabilityProvider),
  `fuel-core-types/src/probe.rs` (BackendId, DeviceDescriptor,
  EquivalenceKey),
  `fuel-core-types/src/device.rs` (DeviceLocation).
- Existing fragmented topology bits to compose:
  `fuel-storage/src/dispatch.rs` (CapabilityRegistry,
  global_registry),
  `fuel-core/src/probe.rs` (ProbeReport::probe_all),
  `fuel-graph-router/src/lib.rs` lines 626-786 (Router's
  capability_index — closest existing parallel).
- Pattern reference: `fuel-storage/src/dispatch.rs::global_bindings`
  is the OnceLock-with-lazy-init shape SystemTopology should
  mirror.
