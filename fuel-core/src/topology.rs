//! SystemTopology — single source of truth for what backends exist,
//! what devices they target, which backends share Storage substrate,
//! and what transfer paths connect devices.
//!
//! See `docs/session-prompts/system-topology-service.md` for the
//! session prompt that scoped this module + the architectural
//! decisions (TDP-1…TDP-7) the implementation resolves.
//!
//! # Design summary
//!
//! - **TDP-1** SystemTopology lives in `fuel-core::topology`. Same
//!   dependency height as `fuel-core::dispatch` (the Judge consumer) —
//!   needs `ProbeReport` (fuel-core), `global_bindings()` +
//!   `global_registry()` (fuel-storage), `DeviceLocation` /
//!   `BackendId` / `SubstrateClass` (fuel-core-types).
//! - **TDP-2** Substrate is encoded as a new
//!   [`fuel_core_types::backend::SubstrateClass`] field on
//!   `BackendCapabilities`. Backends self-declare. SystemTopology
//!   falls back to a sensible per-`BackendId` default for backends
//!   that haven't registered capabilities yet (CUDA/Vulkan today),
//!   so the predicate is correct even before the full per-backend
//!   capability-provider refactor lands.
//! - **TDP-3** `shares_storage` is keyed by `(BackendId,
//!   DeviceLocation)` so CUDA gpu_id=0 vs gpu_id=1 distinguish.
//! - **TDP-4** TransferPath is returned as the enum discriminator.
//!   Numeric cost estimates — deferred at TDP-4 time — shipped as
//!   Stage 1 of the load-time incremental planner
//!   (`docs/session-prompts/load-time-incremental-planner.md`):
//!   [`SystemTopology::transfer_estimate`] /
//!   [`SystemTopology::estimate_transfer_ns`] price every path,
//!   backed by a lazily-probed once-per-generation
//!   [`TransferCalibration`] with conservative per-path-class
//!   fallbacks for unprobed paths.
//! - **TDP-5** Lifecycle uses a generation counter
//!   ([`fuel_dispatch::dispatch::topology_generation`]) plus an
//!   `RwLock<Option<Arc<…>>>`. `current()` rebuilds atomically when
//!   the counter advances; otherwise it returns the cached `Arc`.
//! - **TDP-6** The build re-reads the generation counter *inside*
//!   the build so a labelled-N-but-built-from-N+1 snapshot can't
//!   happen. A stale build is fine (self-healing on next access);
//!   a mislabelled build is not.
//! - **TDP-7** The kernel binding table is the source of truth for
//!   "which backends have kernels registered." Op-coverage advertised
//!   in `BackendCapabilities` is cross-checked against it via
//!   [`SystemTopology::capabilities_op_coverage_is_subset`].

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock, RwLock};

use fuel_core_types::backend::{BackendCapabilities, SubstrateClass, TransferPath};
use fuel_core_types::dispatch::OpKind;
use fuel_core_types::probe::BackendId;
use fuel_core_types::{DType, DeviceLocation};
use fuel_dispatch::dispatch::{global_bindings, global_registry, topology_generation};

use crate::probe::ProbeReport;
use crate::transfer_cost::{TransferCalibration, TransferEstimate};

/// Process-wide cached topology. Holds at most one `Arc<SystemTopology>`
/// at a time; replaced atomically when the generation counter advances.
static CURRENT_TOPOLOGY: RwLock<Option<Arc<SystemTopology>>> = RwLock::new(None);

/// Single-source-of-truth view of which backends are loaded in this
/// process, which devices exist, which backends share storage
/// substrate (so cross-backend on the same device is free), and what
/// transfer paths connect devices.
///
/// Construct via [`SystemTopology::current`] — the result is an
/// `Arc` so a long-running consumer (a picker walking a large graph)
/// has a stable view for the duration of its call even if a backend
/// registers mid-walk.
#[derive(Debug)]
pub struct SystemTopology {
    /// Topology-generation counter the snapshot was built from. If
    /// the live counter has moved past this value, the snapshot is
    /// stale and the next [`current()`](SystemTopology::current)
    /// call rebuilds.
    generation: u64,
    /// All distinct device locations seen by the loaded backends.
    /// Sorted by `BackendId`-grouped order; CPU first.
    devices: Vec<DeviceLocation>,
    /// All distinct `BackendId`s that have at least one kernel in the
    /// global binding table OR an entry in the global capability
    /// registry. Sorted ascending by `as_str()`.
    backends: Vec<BackendId>,
    /// `device → [backend, ...]`. Built from probe + registry.
    /// Order within the value preserves registration / probe order so
    /// callers can use "first" as a deterministic default.
    backends_for_device: HashMap<DeviceLocation, Vec<BackendId>>,
    /// `backend → [device, ...]`. Inverse of the above.
    devices_for_backend: HashMap<BackendId, Vec<DeviceLocation>>,
    /// Substrate class declared by each backend. Populated from
    /// registered `BackendCapabilities::storage_substrate` when the
    /// backend has registered its capabilities, else from
    /// [`default_substrate_for`] as a forward-compatible fallback.
    substrate_for: HashMap<BackendId, SubstrateClass>,
    /// Capability snapshot per backend. Populated only for backends
    /// whose capabilities are in `global_registry()`. Backends that
    /// register kernels into the binding table but never call
    /// `register_backend_capabilities` are absent here — that's a
    /// gap the picker / diagnostics surface, not an error.
    capabilities: HashMap<BackendId, BackendCapabilities>,
    /// Transfer-path matrix `(src, dst) → path`, consolidated from
    /// each registered backend's `transfer_paths`. Missing entries
    /// fall back to [`TransferPath::HostStaging`] (every backend
    /// supports host staging by contract).
    transfer_paths: HashMap<(DeviceLocation, DeviceLocation), TransferPath>,
    /// `backend → set((op, dtype))` derived from the live binding
    /// table. The source of truth for "what kernels exist." A
    /// backend's `BackendCapabilities::op_dtype_support` advertisement
    /// can be cross-checked against this for the TDP-7 divergence
    /// guard.
    binding_op_coverage: HashMap<BackendId, HashSet<(OpKind, DType)>>,
    /// Numeric per-path transfer calibration (planner Stage 1).
    /// Probed lazily on the first [`Self::transfer_calibration`] /
    /// [`Self::transfer_estimate`] call and cached for the lifetime
    /// of this snapshot — i.e. once per topology generation. Empty
    /// (and free to build) on CPU-only hosts.
    transfer_calibration: OnceLock<TransferCalibration>,
}

impl SystemTopology {
    /// Return a snapshot of the current topology. Cheap when nothing
    /// has changed since the last call (one atomic load + `Arc::clone`);
    /// rebuilds + atomically swaps when the
    /// [generation counter](fuel_dispatch::dispatch::topology_generation)
    /// has advanced.
    ///
    /// The returned `Arc` lives independent of the cache lock — a
    /// long-running consumer can hold the snapshot across calls that
    /// might trigger a rebuild without risk of deadlock or torn reads.
    ///
    /// See `docs/session-prompts/system-topology-service.md` for the
    /// full TDP-5/TDP-6 lifecycle contract.
    pub fn current() -> Arc<SystemTopology> {
        let cur_gen = topology_generation();
        // Fast path: cached snapshot is current.
        if let Some(t) = CURRENT_TOPOLOGY.read().unwrap().as_ref() {
            if t.generation == cur_gen {
                return Arc::clone(t);
            }
        }
        // Slow path: rebuild. We may race with another rebuild; the
        // last writer wins and both produce an internally-consistent
        // view (the build re-reads the counter inside `build_at` —
        // see TDP-6).
        let fresh = Arc::new(SystemTopology::build_at(cur_gen));
        let mut guard = CURRENT_TOPOLOGY.write().unwrap();
        // Re-check under the write lock: another thread may have built
        // a newer or equal snapshot already.
        match guard.as_ref() {
            Some(existing) if existing.generation >= fresh.generation => Arc::clone(existing),
            _ => {
                *guard = Some(Arc::clone(&fresh));
                fresh
            }
        }
    }

    /// Force a rebuild on the next [`current()`](Self::current) call,
    /// even if no registration site has bumped the generation
    /// counter. Use in tests / advanced setups where a registration
    /// path bypassed the standard helpers; production code should let
    /// registration sites bump the counter via the
    /// `register_backend_capabilities` / `extend_global_bindings`
    /// helpers in fuel-storage.
    pub fn refresh() {
        fuel_dispatch::dispatch::bump_topology_generation();
    }

    /// Generation counter this snapshot was built against. Two
    /// snapshots with equal generations report identical predicates.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Every distinct [`DeviceLocation`] visible on this host. Sorted
    /// for deterministic iteration; CPU is always present.
    pub fn devices(&self) -> &[DeviceLocation] {
        &self.devices
    }

    /// Every [`BackendId`] with at least one registered kernel or
    /// capability entry. Sorted ascending by `as_str()`.
    pub fn backends(&self) -> &[BackendId] {
        &self.backends
    }

    /// Which backends can target this device? CPU returns `[Cpu,
    /// Reference, Aocl, Mkl]` (subset of compiled-in backends);
    /// `Cuda { gpu_id: N }` returns `[Cuda]`; `Vulkan { gpu_id: N }`
    /// returns `[Vulkan]`. Empty slice for an unknown device.
    pub fn backends_for(&self, dev: DeviceLocation) -> &[BackendId] {
        self.backends_for_device
            .get(&dev)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Which devices can this backend target? CUDA returns every
    /// `Cuda { gpu_id }` visible on the host. Empty slice for an
    /// unknown backend or one whose probe returned zero devices.
    pub fn devices_for(&self, backend: BackendId) -> &[DeviceLocation] {
        self.devices_for_backend
            .get(&backend)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// The substrate class for a `(backend, device)` pair. Returns
    /// `None` if the backend is unknown.
    ///
    /// Prefers the substrate declared in the backend's
    /// [`BackendCapabilities`] when one is registered, otherwise
    /// falls back to [`default_substrate_for`]. The `device` argument
    /// is accepted for API symmetry with [`Self::shares_storage`]
    /// (today's substrate classification doesn't vary per-device but
    /// future NUMA-split CPU might).
    pub fn substrate_class(
        &self,
        backend: BackendId,
        _device: DeviceLocation,
    ) -> Option<SubstrateClass> {
        self.substrate_for.get(&backend).copied()
    }

    /// **Critical predicate** — do these two backends operate on the
    /// same storage substrate when both target their given device?
    ///
    /// `shares_storage((Cpu, Cpu), (Aocl, Cpu))` is true — both
    /// declare [`SubstrateClass::HostBytes`] and run on
    /// `DeviceLocation::Cpu`, so a kernel from either backend can
    /// consume the other's output with no copy.
    ///
    /// `shares_storage((Cuda, Cuda{0}), (Cuda, Cuda{1}))` is false —
    /// same substrate class, different devices: a peer or
    /// host-staging copy is required (see [`Self::transfer_path`]).
    ///
    /// `shares_storage((Cuda, Cuda{0}), (Vulkan, Vulkan{0}))` is
    /// false — same physical silicon, but the substrates are
    /// distinct (CUDA's allocator vs Vulkan's `VkBuffer`). External-
    /// memory import is out of scope today; treat as host-staging.
    ///
    /// Returns false if either backend is unknown.
    pub fn shares_storage(
        &self,
        a: (BackendId, DeviceLocation),
        b: (BackendId, DeviceLocation),
    ) -> bool {
        let (a_backend, a_device) = a;
        let (b_backend, b_device) = b;
        if a_device != b_device {
            return false;
        }
        match (
            self.substrate_for.get(&a_backend),
            self.substrate_for.get(&b_backend),
        ) {
            (Some(sa), Some(sb)) => sa == sb,
            _ => false,
        }
    }

    /// What's needed to move bytes from `src` to `dst`? Returns the
    /// path advertised by the source backend's capabilities, or
    /// [`TransferPath::SameDevice`] when `src == dst`, or
    /// [`TransferPath::HostStaging`] as the universal fallback when
    /// no specific path was advertised (every backend supports host
    /// staging by contract).
    pub fn transfer_path(&self, src: DeviceLocation, dst: DeviceLocation) -> TransferPath {
        if src == dst {
            return TransferPath::SameDevice;
        }
        self.transfer_paths
            .get(&(src, dst))
            .copied()
            .unwrap_or(TransferPath::HostStaging)
    }

    /// Numeric per-path transfer calibration — Stage 1 of the
    /// load-time incremental planner
    /// (`docs/session-prompts/load-time-incremental-planner.md`).
    ///
    /// The first call on a snapshot runs the calibration probe
    /// (timed H2D/D2H round-trips through the byte-storage substrate
    /// APIs, a few tens of ms per GPU device); every later call —
    /// from any thread — returns the cached result. Because the
    /// snapshot itself is rebuilt at most once per topology
    /// generation, the probe runs at most once per generation.
    /// CPU-only hosts probe zero paths and return an empty
    /// calibration immediately — no probe, no error.
    ///
    /// Most consumers want [`Self::transfer_estimate`] /
    /// [`Self::estimate_transfer_ns`], which add the fallback
    /// pricing for unprobed paths; this accessor exposes the raw
    /// probed set for diagnostics and tests.
    pub fn transfer_calibration(&self) -> &TransferCalibration {
        self.transfer_calibration
            .get_or_init(|| TransferCalibration::calibrate(&self.devices))
    }

    /// Numeric cost model for moving bytes `src → dst`, for the
    /// planner's cost composer (Stage 2) and placement DP (Stage 3).
    /// Never fails:
    ///
    /// 1. a probed estimate for the exact path wins;
    /// 2. an unprobed device-to-device pair whose two host legs were
    ///    both probed prices as their serial composition (today's
    ///    cross-GPU transfers route through host staging — see
    ///    [`Self::transfer_path`]);
    /// 3. anything else uses the conservative per-path-class static
    ///    defaults documented on [`TransferEstimate::fallback_for`].
    ///
    /// Triggers the lazy calibration probe on first use (see
    /// [`Self::transfer_calibration`]).
    pub fn transfer_estimate(&self, src: DeviceLocation, dst: DeviceLocation) -> TransferEstimate {
        if src == dst {
            return TransferEstimate::fallback_for(TransferPath::SameDevice);
        }
        let cal = self.transfer_calibration();
        if let Some(e) = cal.probed(src, dst) {
            return e;
        }
        if src != DeviceLocation::Cpu && dst != DeviceLocation::Cpu {
            if let (Some(d2h), Some(h2d)) = (
                cal.probed(src, DeviceLocation::Cpu),
                cal.probed(DeviceLocation::Cpu, dst),
            ) {
                return TransferEstimate::compose_staged(d2h, h2d);
            }
        }
        TransferEstimate::fallback_for(self.transfer_path(src, dst))
    }

    /// Estimated wall-clock nanoseconds to move `bytes` from `src`
    /// to `dst`. Zero when `src == dst` (no transfer needed).
    /// Saturating, monotonic in `bytes`, and never panics — unprobed
    /// paths use the conservative fallbacks, never a garbage zero.
    pub fn estimate_transfer_ns(
        &self,
        src: DeviceLocation,
        dst: DeviceLocation,
        bytes: u64,
    ) -> u64 {
        if src == dst {
            return 0;
        }
        self.transfer_estimate(src, dst).estimate_ns(bytes)
    }

    /// Per-backend [`BackendCapabilities`] snapshot if the backend has
    /// registered with [`fuel_dispatch::dispatch::register_backend_capabilities`].
    /// Backends that only registered kernels into the binding table
    /// (most production paths today) return `None`. The picker should
    /// not assume capabilities are present; it can fall back to the
    /// binding-table walk via [`Self::binding_op_coverage`].
    pub fn capabilities(&self, backend: BackendId) -> Option<&BackendCapabilities> {
        self.capabilities.get(&backend)
    }

    /// `(op, dtype)` pairs derived from the live binding table for a
    /// backend — the source of truth for "what kernels exist." Empty
    /// set if the backend has no registered kernels. Note that the
    /// binding table key includes a per-operand dtype list; this
    /// flattens it to the output dtype (the last in the list) for the
    /// classic `(op, dtype)` shape consumers expect.
    pub fn binding_op_coverage(&self, backend: BackendId) -> &HashSet<(OpKind, DType)> {
        static EMPTY: std::sync::OnceLock<HashSet<(OpKind, DType)>> = std::sync::OnceLock::new();
        self.binding_op_coverage
            .get(&backend)
            .unwrap_or_else(|| EMPTY.get_or_init(HashSet::new))
    }

    /// TDP-7 divergence guard: for every backend that advertises an
    /// `op_dtype_support` set in [`BackendCapabilities`], assert
    /// every advertised pair has a corresponding entry in the live
    /// binding table. Returns the list of `(backend, op, dtype)`
    /// pairs that were advertised but not registered. Empty list =
    /// no divergence. Used by the topology divergence test.
    pub fn capabilities_op_coverage_divergence(
        &self,
    ) -> Vec<(BackendId, OpKind, DType)> {
        let mut missing = Vec::new();
        for (backend, caps) in &self.capabilities {
            let registered = self.binding_op_coverage(*backend);
            for &(op, dtype) in &caps.op_dtype_support {
                if !registered.contains(&(op, dtype)) {
                    missing.push((*backend, op, dtype));
                }
            }
        }
        missing.sort_by(|a, b| {
            a.0.as_str().cmp(b.0.as_str())
                .then_with(|| format!("{:?}", a.1).cmp(&format!("{:?}", b.1)))
        });
        missing
    }

    /// Build a fresh topology snapshot, reading the generation
    /// counter *inside* the build so the resulting snapshot is
    /// labelled with the generation it actually reflects (TDP-6). A
    /// concurrent registration that lands mid-build bumps the counter
    /// further; the next `current()` call observes the higher value
    /// and rebuilds again — self-healing.
    fn build_at(_caller_gen: u64) -> SystemTopology {
        let built_gen = topology_generation();

        // Device enumeration: every compiled-in backend's probe.
        // Reference + CPU are always present; other backends gate
        // on cargo features.
        let probe = ProbeReport::probe_all();

        // Per-backend capability snapshots from the global registry.
        // Today only CPU is auto-registered; future backends will
        // join here when their capability-provider impls land.
        let mut capabilities: HashMap<BackendId, BackendCapabilities> = HashMap::new();
        {
            let registry = global_registry();
            for caps in registry.backends() {
                // First-wins: matches CapabilityRegistry's lookup
                // convention where the first-registered backend for
                // a `(op, dtype)` wins ties.
                capabilities.entry(caps.backend_id).or_insert_with(|| caps.clone());
            }
        }

        // Binding-table op-coverage per backend — source of truth for
        // "which backends have kernels registered."
        let mut binding_op_coverage: HashMap<BackendId, HashSet<(OpKind, DType)>> = HashMap::new();
        {
            let bindings = global_bindings();
            for (op, dtypes, backend) in bindings.iter_keys() {
                // The binding key carries per-operand dtypes (inputs +
                // outputs). The classic `(op, dtype)` shape pulls the
                // output dtype — the last entry in the list. Single-
                // dtype keys (most CPU ops) are unambiguous; multi-
                // dtype keys still produce a sensible entry (the
                // output is what consumers care about).
                if let Some(&output_dt) = dtypes.last() {
                    binding_op_coverage
                        .entry(backend)
                        .or_default()
                        .insert((op, output_dt));
                }
            }
        }

        // The union of backends — present if they have at least one
        // registered kernel OR a capability entry. Sorted for
        // determinism.
        let mut backends: Vec<BackendId> = capabilities
            .keys()
            .copied()
            .chain(binding_op_coverage.keys().copied())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        backends.sort_by(|a, b| a.as_str().cmp(b.as_str()));

        // Per-backend substrate class. Prefer the registered
        // declaration; otherwise fall back to the per-BackendId
        // default. The fallback exists because CUDA / Vulkan etc.
        // today register only kernels, not capabilities — they'll
        // start declaring substrate explicitly as the per-backend
        // capability-provider refactor lands.
        let mut substrate_for: HashMap<BackendId, SubstrateClass> = HashMap::new();
        for &b in &backends {
            let cls = capabilities
                .get(&b)
                .map(|c| c.storage_substrate)
                .unwrap_or_else(|| default_substrate_for(b));
            substrate_for.insert(b, cls);
        }

        // Device set: union of probe-reported devices, registered
        // BackendCapabilities device_locations, and a synthetic CPU
        // entry to guarantee CPU is always present.
        let mut device_set: HashSet<DeviceLocation> = HashSet::new();
        device_set.insert(DeviceLocation::Cpu);
        for d in &probe.devices {
            device_set.insert(d.location);
        }
        for caps in capabilities.values() {
            device_set.insert(caps.device_location);
        }
        // Synthesise a default device for any backend that has
        // kernels but no probe entry (defensive — every real backend
        // probe returns at least one device on a host that loaded
        // its runtime, but tests with mock backends may skip probe
        // entirely).
        for &b in &backends {
            device_set.insert(default_device_for(b));
        }
        // Stable order: CPU first, then sort by `Debug` repr.
        let mut devices: Vec<DeviceLocation> = device_set.into_iter().collect();
        devices.sort_by(|a, b| match (a, b) {
            (DeviceLocation::Cpu, DeviceLocation::Cpu) => std::cmp::Ordering::Equal,
            (DeviceLocation::Cpu, _) => std::cmp::Ordering::Less,
            (_, DeviceLocation::Cpu) => std::cmp::Ordering::Greater,
            (x, y) => format!("{x:?}").cmp(&format!("{y:?}")),
        });

        // `device → [backend]` and inverse.
        let mut backends_for_device: HashMap<DeviceLocation, Vec<BackendId>> = HashMap::new();
        let mut devices_for_backend: HashMap<BackendId, Vec<DeviceLocation>> = HashMap::new();

        // First seed from probe — authoritative for which devices a
        // backend can actually reach on this host.
        let mut probe_devices_by_backend: HashMap<BackendId, Vec<DeviceLocation>> = HashMap::new();
        for d in &probe.devices {
            probe_devices_by_backend
                .entry(d.backend)
                .or_default()
                .push(d.location);
        }
        for &b in &backends {
            let probe_devs = probe_devices_by_backend.remove(&b);
            let bd_devices = match probe_devs {
                Some(v) if !v.is_empty() => v,
                _ => {
                    // No probe entries for this backend (mock /
                    // incomplete probe). Synthesise a default-device
                    // entry so the predicates still return sane
                    // answers.
                    if let Some(caps) = capabilities.get(&b) {
                        vec![caps.device_location]
                    } else {
                        vec![default_device_for(b)]
                    }
                }
            };
            for &dev in &bd_devices {
                backends_for_device.entry(dev).or_default().push(b);
            }
            devices_for_backend.insert(b, bd_devices);
        }

        // Ensure every `devices` entry has at least an empty bucket
        // in `backends_for_device` so iteration is symmetric.
        for &dev in &devices {
            backends_for_device.entry(dev).or_default();
        }

        // Transfer-path matrix: union of every backend's advertised
        // outbound transfer_paths. Backends that haven't registered
        // capabilities contribute nothing here; `transfer_path()`
        // falls back to HostStaging in that case.
        let mut transfer_paths: HashMap<(DeviceLocation, DeviceLocation), TransferPath> =
            HashMap::new();
        for caps in capabilities.values() {
            for (dst, path) in &caps.transfer_paths {
                transfer_paths.insert((caps.device_location, *dst), *path);
            }
        }

        SystemTopology {
            generation: built_gen,
            devices,
            backends,
            backends_for_device,
            devices_for_backend,
            substrate_for,
            capabilities,
            transfer_paths,
            binding_op_coverage,
            transfer_calibration: OnceLock::new(),
        }
    }
}

/// Planner Stage-2 adapter: `SystemTopology` IS the production
/// [`fuel_dispatch::ranker::TransferEstimator`]. The optimizer
/// ranker lives in `fuel-dispatch`, which must not depend on
/// `fuel-core` (dependency direction), so `compile_plan` consumes
/// transfer pricing through the trait — threaded via
/// `PlanOptions::with_transfer_estimator` in
/// `pipelined_bridge::build_execution_plan` — while the numbers
/// come from the Stage-1 [`TransferCalibration`] behind
/// [`SystemTopology::estimate_transfer_ns`] (probed lazily once per
/// generation; conservative per-path-class fallbacks otherwise).
/// Same-device queries short-circuit to zero WITHOUT touching the
/// calibration, so CPU-only plans never trigger a probe.
impl fuel_dispatch::ranker::TransferEstimator for SystemTopology {
    fn estimate_transfer_ns(
        &self,
        src: DeviceLocation,
        dst: DeviceLocation,
        bytes: u64,
    ) -> u64 {
        // Inherent method (same name) takes resolution priority —
        // no recursion.
        SystemTopology::estimate_transfer_ns(self, src, dst, bytes)
    }
}

/// Default substrate class for a backend. Used when a backend
/// hasn't registered its [`BackendCapabilities`] yet (so we don't
/// have an explicit declaration). Stays in lockstep with what each
/// backend's storage type is: CPU-trio backends produce
/// `CpuStorageBytes`, CUDA produces `CudaStorageBytes`, Vulkan
/// produces `VulkanStorageBytes`. The future per-backend
/// capability-provider refactor will let backends declare this
/// explicitly and the fallback becomes dead code — but until then
/// the fallback keeps `shares_storage` correct.
fn default_substrate_for(backend: BackendId) -> SubstrateClass {
    match backend {
        BackendId::Cpu => SubstrateClass::HostBytes,
        BackendId::Cuda => SubstrateClass::CudaUntyped,
        BackendId::Vulkan => SubstrateClass::VulkanBuffer,
        BackendId::Metal => SubstrateClass::MetalBuffer,
        // `BackendId` is `#[non_exhaustive]`. A new BackendId variant
        // added downstream defaults to HostBytes; the right
        // long-term answer is for the new backend to declare its
        // substrate explicitly in its `BackendCapabilities`.
        _ => SubstrateClass::HostBytes,
    }
}

/// Default device for a backend when probe data is unavailable. Real
/// hosts will have probe-reported devices; this fallback exists for
/// tests with mock backends and for the synthetic-fallback path in
/// `build_at`.
fn default_device_for(backend: BackendId) -> DeviceLocation {
    match backend {
        BackendId::Cpu => DeviceLocation::Cpu,
        BackendId::Cuda => DeviceLocation::Cuda { gpu_id: 0 },
        BackendId::Vulkan => DeviceLocation::Vulkan { gpu_id: 0 },
        BackendId::Metal => DeviceLocation::Metal { gpu_id: 0 },
        // `BackendId` is `#[non_exhaustive]`; an unknown new variant
        // defaults to CPU. Real backends should declare their
        // device_location in `BackendCapabilities` so the topology
        // gets the right device without consulting this fallback.
        _ => DeviceLocation::Cpu,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Always-on baseline: CPU is in the topology regardless of which
    /// optional backends are compiled in.
    #[test]
    fn cpu_always_present() {
        let topology = SystemTopology::current();
        assert!(topology.devices().contains(&DeviceLocation::Cpu));
        assert!(topology.backends().contains(&BackendId::Cpu));
        let backends_at_cpu = topology.backends_for(DeviceLocation::Cpu);
        assert!(
            backends_at_cpu.contains(&BackendId::Cpu),
            "Cpu should be among the backends at DeviceLocation::Cpu, got: {:?}",
            backends_at_cpu,
        );
    }

    /// Substrate predicate: CPU shares with itself; HostBytes-class
    /// backends share at DeviceLocation::Cpu.
    #[test]
    fn cpu_shares_with_itself() {
        let topology = SystemTopology::current();
        assert!(topology.shares_storage(
            (BackendId::Cpu, DeviceLocation::Cpu),
            (BackendId::Cpu, DeviceLocation::Cpu),
        ));
    }

    /// Transfer path: same device → SameDevice; missing entry →
    /// HostStaging fallback.
    #[test]
    fn transfer_path_defaults() {
        let topology = SystemTopology::current();
        assert_eq!(
            topology.transfer_path(DeviceLocation::Cpu, DeviceLocation::Cpu),
            TransferPath::SameDevice,
        );
    }

    /// Stage 1 cache contract: the calibration probe runs at most
    /// once per snapshot (and snapshots rebuild at most once per
    /// generation) — two calls on the same snapshot return the same
    /// cached instance.
    #[test]
    fn transfer_calibration_cached_per_snapshot() {
        let topology = SystemTopology::current();
        let a = topology.transfer_calibration();
        let b = topology.transfer_calibration();
        assert!(
            std::ptr::eq(a, b),
            "transfer_calibration must be probed once and cached on the snapshot",
        );
        // A fresh generation gets a fresh snapshot and therefore a
        // fresh (lazy) calibration slot.
        SystemTopology::refresh();
        let next = SystemTopology::current();
        assert!(
            !std::ptr::eq(a, next.transfer_calibration()),
            "a new generation's snapshot must carry its own calibration",
        );
    }

    /// Same-device moves cost zero; cross-device estimates are never
    /// garbage-zero and grow monotonically with bytes.
    #[test]
    fn transfer_estimate_same_device_zero_and_fallback_monotonic() {
        let topology = SystemTopology::current();
        assert_eq!(
            topology.estimate_transfer_ns(DeviceLocation::Cpu, DeviceLocation::Cpu, 64 << 20),
            0,
        );
        // gpu_id 7 exists on no probe report — this pair is
        // guaranteed unprobed in every build configuration, so it
        // exercises the HostStaging fallback deterministically.
        let phantom = DeviceLocation::Cuda { gpu_id: 7 };
        let est = topology.transfer_estimate(DeviceLocation::Cpu, phantom);
        assert_eq!(
            est,
            crate::transfer_cost::TransferEstimate::fallback_for(TransferPath::HostStaging),
            "unprobed path must price via the per-path-class fallback",
        );
        let small = topology.estimate_transfer_ns(DeviceLocation::Cpu, phantom, 64 * 1024);
        let large = topology.estimate_transfer_ns(DeviceLocation::Cpu, phantom, 64 << 20);
        assert!(small > 0, "unprobed cross-device transfer must never cost zero");
        assert!(large > small, "estimate must be monotonic in bytes");
    }

    /// Live RTX 4070 calibration over the CUDA byte-storage path.
    /// Run: `cargo test -p fuel-core --lib --features cuda
    /// topology::tests::live_cuda_transfer_calibration -- --ignored`
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a live CUDA GPU"]
    fn live_cuda_transfer_calibration_sane() {
        let topology = SystemTopology::current();
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        assert!(
            topology.devices().contains(&cuda0),
            "live test expects a CUDA device, got {:?}",
            topology.devices(),
        );
        let cal = topology.transfer_calibration();
        let h2d = cal
            .probed(DeviceLocation::Cpu, cuda0)
            .expect("H2D path must be probed on a CUDA host");
        let d2h = cal
            .probed(cuda0, DeviceLocation::Cpu)
            .expect("D2H path must be probed on a CUDA host");
        const GB: u64 = 1_000_000_000;
        for (label, est) in [("H2D", h2d), ("D2H", d2h)] {
            assert!(
                (GB..=100 * GB).contains(&est.bandwidth_bytes_per_sec),
                "{label} bandwidth out of sane bounds (1-100 GB/s): {est:?}",
            );
        }
        // estimate_ns monotonic across the calibration sizes, and
        // strictly larger for a 1000× larger transfer.
        let est = topology.transfer_estimate(DeviceLocation::Cpu, cuda0);
        let ns_small = est.estimate_ns(64 * 1024);
        let ns_mid = est.estimate_ns(4 << 20);
        let ns_large = est.estimate_ns(64 << 20);
        assert!(ns_small <= ns_mid && ns_mid <= ns_large);
        assert!(ns_large > ns_small, "64 MB must cost more than 64 KB");
    }

    /// Live Vulkan sibling of the CUDA calibration test, with
    /// direction-asymmetric bounds reflecting what the byte-storage
    /// path actually costs:
    ///
    /// - **H2D** is write-combined host→staging plus a device copy —
    ///   GB/s-class (0.2–100 GB/s bound; staging-buffer creation +
    ///   a fenced one-shot submit per call keep it below CUDA's).
    /// - **D2H** reads back through a host-visible mapped pointer
    ///   that on NVIDIA Windows is typically *not* HOST_CACHED, so
    ///   the CPU-side copy runs at uncached-read speeds — measured
    ///   ~140 MB/s on the RTX 4070 box. That slowness is real (it's
    ///   what the planner must price), so the bound only guards
    ///   against garbage: 10 MB/s–100 GB/s.
    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires a live Vulkan GPU"]
    fn live_vulkan_transfer_calibration_sane() {
        let topology = SystemTopology::current();
        let vk0 = DeviceLocation::Vulkan { gpu_id: 0 };
        assert!(
            topology.devices().contains(&vk0),
            "live test expects a Vulkan device, got {:?}",
            topology.devices(),
        );
        let cal = topology.transfer_calibration();
        let h2d = cal
            .probed(DeviceLocation::Cpu, vk0)
            .expect("H2D path must be probed on a Vulkan host");
        let d2h = cal
            .probed(vk0, DeviceLocation::Cpu)
            .expect("D2H path must be probed on a Vulkan host");
        const GB: u64 = 1_000_000_000;
        assert!(
            (GB / 5..=100 * GB).contains(&h2d.bandwidth_bytes_per_sec),
            "H2D bandwidth out of sane bounds (0.2-100 GB/s): {h2d:?}",
        );
        assert!(
            (10_000_000..=100 * GB).contains(&d2h.bandwidth_bytes_per_sec),
            "D2H bandwidth out of sane bounds (10 MB/s-100 GB/s): {d2h:?}",
        );
        let est = topology.transfer_estimate(DeviceLocation::Cpu, vk0);
        assert!(est.estimate_ns(64 << 20) > est.estimate_ns(64 * 1024));
    }

    /// CPU's BackendCapabilities is always registered, so the
    /// capabilities lookup must succeed.
    #[test]
    fn cpu_capabilities_present() {
        let topology = SystemTopology::current();
        let caps = topology
            .capabilities(BackendId::Cpu)
            .expect("CPU caps always registered via default_cpu_caps");
        assert_eq!(caps.backend_id, BackendId::Cpu);
        assert_eq!(caps.storage_substrate, SubstrateClass::HostBytes);
    }

    /// TDP-7 divergence guard: every (op, dtype) the CPU backend
    /// advertises in its op_dtype_support must have a corresponding
    /// binding-table entry.
    #[test]
    fn cpu_op_coverage_no_divergence() {
        let topology = SystemTopology::current();
        let divergence = topology.capabilities_op_coverage_divergence();
        let cpu_divergence: Vec<_> = divergence
            .iter()
            .filter(|(b, _, _)| *b == BackendId::Cpu)
            .collect();
        assert!(
            cpu_divergence.is_empty(),
            "CPU advertises ops it didn't register: {:#?}",
            cpu_divergence,
        );
    }

    /// Live-update: a fresh bump invalidates the cached snapshot.
    /// Two `current()` calls with no intervening change return the
    /// same Arc. Because the topology cache is process-wide and
    /// parallel tests may bump the generation, we loop until we
    /// observe two consecutive calls where the generation didn't
    /// advance, then assert Arc identity over that window.
    #[test]
    fn no_change_reuses_arc() {
        use fuel_dispatch::dispatch::topology_generation;
        for _ in 0..32 {
            let gen_before = topology_generation();
            let a = SystemTopology::current();
            let b = SystemTopology::current();
            let gen_after = topology_generation();
            if gen_before == gen_after && a.generation() == b.generation() {
                assert!(
                    Arc::ptr_eq(&a, &b),
                    "two current() calls with no intervening generation \
                     change should return the same Arc (gen={})",
                    gen_before,
                );
                return;
            }
        }
        panic!(
            "Could not observe a stable-generation window across 32 \
             attempts — parallel tests may be saturating the counter; \
             rerun with --test-threads=1 if this persists",
        );
    }

    /// Live-update: bumping the generation forces a rebuild.
    #[test]
    fn bump_forces_rebuild() {
        let before = SystemTopology::current();
        SystemTopology::refresh();
        let after = SystemTopology::current();
        assert!(
            after.generation() > before.generation(),
            "refresh() should advance the generation: before={}, after={}",
            before.generation(),
            after.generation(),
        );
        // The two snapshots are distinct Arc allocations after the
        // rebuild — the cache was swapped, not mutated.
        assert!(
            !Arc::ptr_eq(&before, &after),
            "refresh() should produce a fresh Arc",
        );
    }

    /// Concurrent access: spawning N threads that hammer current()
    /// while another bumps the generation must not panic, and every
    /// snapshot must answer its predicates consistently with its
    /// reported generation.
    #[test]
    fn concurrent_access_is_safe() {
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicBool, AtomicUsize};
        use std::thread;
        use std::time::Duration;

        let stop = StdArc::new(AtomicBool::new(false));
        let max_backends = StdArc::new(AtomicUsize::new(0));

        let mut readers = Vec::new();
        for _ in 0..8 {
            let stop = StdArc::clone(&stop);
            let max_backends = StdArc::clone(&max_backends);
            readers.push(thread::spawn(move || {
                while !stop.load(std::sync::atomic::Ordering::Acquire) {
                    let t = SystemTopology::current();
                    // Consistency check: every reported backend must
                    // resolve in devices_for_backend.
                    for &b in t.backends() {
                        let devs = t.devices_for(b);
                        assert!(
                            !devs.is_empty(),
                            "backend {:?} has no devices in snapshot gen={}",
                            b, t.generation(),
                        );
                    }
                    let n = t.backends().len();
                    let mut cur = max_backends.load(std::sync::atomic::Ordering::Acquire);
                    while n > cur {
                        match max_backends.compare_exchange_weak(
                            cur,
                            n,
                            std::sync::atomic::Ordering::AcqRel,
                            std::sync::atomic::Ordering::Acquire,
                        ) {
                            Ok(_) => break,
                            Err(prev) => cur = prev,
                        }
                    }
                }
            }));
        }

        // Bump the generation periodically from another thread.
        let bumper_stop = StdArc::clone(&stop);
        let bumper = thread::spawn(move || {
            for _ in 0..50 {
                if bumper_stop.load(std::sync::atomic::Ordering::Acquire) {
                    return;
                }
                SystemTopology::refresh();
                thread::sleep(Duration::from_micros(50));
            }
        });

        bumper.join().expect("bumper thread");
        stop.store(true, std::sync::atomic::Ordering::Release);
        for r in readers {
            r.join().expect("reader thread");
        }
    }

    /// CUDA cfg-gated check: when the cuda feature is on, the
    /// topology must report Cuda as a backend and CUDA's device must
    /// be in `backends_for(Cuda { gpu_id: 0 })`. We don't assume the
    /// host actually has a CUDA-capable GPU — the binding-table walk
    /// is what surfaces the backend (kernels are auto-registered at
    /// `global_bindings()` init unconditionally on the cuda feature).
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_backend_discovered_when_feature_enabled() {
        let topology = SystemTopology::current();
        assert!(
            topology.backends().contains(&BackendId::Cuda),
            "cuda feature is on; topology should report Cuda. Got: {:?}",
            topology.backends(),
        );
        // CUDA's default device gets a slot in backends_for even if
        // probe found nothing (synthetic fallback).
        let cuda_dev = DeviceLocation::Cuda { gpu_id: 0 };
        let here = topology.backends_for(cuda_dev);
        assert!(
            here.contains(&BackendId::Cuda),
            "Cuda should be among the backends targeting {:?}, got {:?}",
            cuda_dev, here,
        );
    }

    /// Vulkan cfg-gated counterpart of the CUDA test.
    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_backend_discovered_when_feature_enabled() {
        let topology = SystemTopology::current();
        assert!(
            topology.backends().contains(&BackendId::Vulkan),
            "vulkan feature is on; topology should report Vulkan. Got: {:?}",
            topology.backends(),
        );
        let vk_dev = DeviceLocation::Vulkan { gpu_id: 0 };
        let here = topology.backends_for(vk_dev);
        assert!(
            here.contains(&BackendId::Vulkan),
            "Vulkan should be among the backends targeting {:?}, got {:?}",
            vk_dev, here,
        );
    }

    /// CUDA + Vulkan on the same physical GPU don't share storage —
    /// CUDA pointers and Vulkan buffers live in distinct allocators
    /// even when they target the same silicon. External-memory
    /// import is deliberately out of scope (see session prompt).
    #[cfg(all(feature = "cuda", feature = "vulkan"))]
    #[test]
    fn cuda_and_vulkan_dont_share_storage() {
        let topology = SystemTopology::current();
        let shares = topology.shares_storage(
            (BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 }),
            (BackendId::Vulkan, DeviceLocation::Vulkan { gpu_id: 0 }),
        );
        assert!(
            !shares,
            "CUDA and Vulkan on the same silicon must not share storage substrate",
        );
        let path = topology.transfer_path(
            DeviceLocation::Cuda { gpu_id: 0 },
            DeviceLocation::Vulkan { gpu_id: 0 },
        );
        // External-memory import is out of scope; expect HostStaging
        // (the universal fallback) unless someone advertises a more
        // specific path — they shouldn't today.
        assert_eq!(
            path,
            TransferPath::HostStaging,
            "Cross-vendor GPU transfer should fall back to host-staging today",
        );
    }
}
