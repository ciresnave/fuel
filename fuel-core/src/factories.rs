//! Backend factory registry — Step 12 of the backend-agnostic refactor.
//!
//! Every backend that fuel-core can drive at runtime declares a single
//! [`BackendFactory`] value in this module. The registry ([`registry`])
//! returns the cfg-gated subset that's actually compiled in, and
//! consumers — currently the [`crate::probe`] enumerator and the
//! [`crate::judge`] profiler — walk that registry instead of naming
//! `fuel_cuda_backend::CudaDevice`/`fuel_vulkan_backend::VulkanBackend`/...
//! by hand.
//!
//! # Realization path (executor-unification Session 2, 2026-06-11)
//!
//! [`LazyRealizer`] realizes a [`crate::lazy::LazyTensor`] through the
//! pipelined bridge ([`crate::pipelined_bridge::realize_one_as_with_initial`])
//! on a pinned [`crate::Device`] — the SAME dispatch path production
//! `realize_f32` uses. Pre-Session-2 this module constructed a typed
//! legacy executor per backend; that was the last architectural
//! reason to retain the 33-method legacy backend trait (re-audit
//! gap 11), and it measured the legacy evaluator rather than the
//! binding-table kernels the picker actually dispatches.
//!
//! The realizer keeps a persistent [`StorageCache`] across calls so
//! `Op::Const` uploads amortize over the Judge's warmup + timed
//! iterations — parity with the retired legacy executor's
//! `const_pool` (without it, every GPU timing iteration would pay
//! H2D for its inputs and the profile would measure PCIe, not the
//! kernel).
//!
//! # Adding a new backend
//!
//! 1. Add a unit struct `MyBackendFactory` here, behind a cfg(feature).
//! 2. Implement `BackendFactory` — `enumerate_devices` delegates to the
//!    backend crate's existing `probe::enumerate_devices`, and
//!    `try_make_realizer` constructs the backend's `crate::Device`
//!    handle and wraps it in [`BridgeRealizer`].
//! 3. Add a cfg-gated entry in [`registry`].
//!
//! No edits to judge.rs or probe.rs.

use crate::lazy::LazyTensor;
use fuel_core_types::probe::{BackendId, DeviceDescriptor};
use fuel_core_types::{Error, Result};
use fuel_dispatch::pipelined::StorageCache;

/// Object-safe realize seam used by judge.rs. Realizes the tensor's
/// graph on the realizer's pinned device through the pipelined
/// bridge and returns the result's host bytes as `Vec<f32>`.
///
/// Result-returning (no-panics policy): a backend that can't realize
/// the graph (missing kernel, device error) surfaces a typed `Err`;
/// the Judge logs and skips the cell.
pub trait LazyRealizer {
    fn realize_f32(&mut self, tensor: &LazyTensor) -> Result<Vec<f32>>;

    /// `kernel_source` of the alternative the picker dispatched for
    /// the most recent [`Self::realize_f32`]'s root node (Session 3
    /// rider — the Judge tags the realizer-measured `CellRun` with
    /// this so multi-sibling cells record the TRUE dispatched
    /// sibling).
    ///
    /// `None` means "no report": either no realize has run yet, or
    /// the plan carried no `AlternativeSet` for the root (the
    /// executor then dispatched the first-registered binding — the
    /// caller's fallback attribution should match that convention).
    /// Defaulted so test stubs without a picker stay one-method.
    fn last_kernel_source(&self) -> Option<&'static str> {
        None
    }
}

/// Bridge-backed realizer pinned to one [`crate::Device`].
///
/// `cache` persists device-resident `Op::Const` storages across
/// calls: the first realize uploads every reachable Const to the
/// pinned device (via [`crate::pipelined_bridge::build_const_cache`]);
/// subsequent calls find the NodeIds already present and skip the
/// upload, so timed iterations measure dispatch + kernel + result
/// download — not input H2D.
struct BridgeRealizer {
    device: crate::Device,
    cache: StorageCache,
    /// Picker attribution from the most recent realize — see
    /// [`LazyRealizer::last_kernel_source`].
    last_kernel_source: Option<&'static str>,
}

impl BridgeRealizer {
    fn new(device: crate::Device) -> Self {
        Self { device, cache: StorageCache::new(), last_kernel_source: None }
    }
}

impl LazyRealizer for BridgeRealizer {
    fn realize_f32(&mut self, tensor: &LazyTensor) -> Result<Vec<f32>> {
        let graph = tensor.graph_tensor().graph().clone();
        let target = tensor.graph_tensor().id();

        // Top up the persistent cache with any reachable Const not
        // yet uploaded. No-op after the first call for a given
        // tensor (build_const_cache skips ids already present).
        let order = {
            let g = graph
                .read()
                .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
            fuel_graph::topo_order_multi(&g, &[target])
        };
        self.cache = crate::pipelined_bridge::build_const_cache(
            &graph,
            &order,
            &self.device,
            std::mem::take(&mut self.cache),
        )?;

        // Planner Stage 2 (2026-06-11): the Judge measures a
        // SPECIFIC backend at a SPECIFIC device, so hard-pin every
        // reachable node with an explicit placement. The priced
        // off-device admission relax only applies to soft
        // (realize-call) pins — without this stamp the planner
        // could legitimately move a profiled op to a "cheaper"
        // sibling device and the cell would record a mislabeled
        // latency. Idempotent across the warmup + timed re-realizes
        // of one cell (same graph, same device).
        {
            let mut g = graph
                .write()
                .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
            let loc = self.device.location();
            for &id in &order {
                g.set_placement(id, loc);
            }
        }

        let (bytes, root_kernel_source) =
            crate::pipelined_bridge::realize_one_as_with_initial_reporting::<f32>(
                &graph,
                target,
                &self.device,
                self.cache.clone(),
            )?;
        self.last_kernel_source = root_kernel_source;
        Ok(bytes)
    }

    fn last_kernel_source(&self) -> Option<&'static str> {
        self.last_kernel_source
    }
}

/// One concrete backend the runtime can drive. Implementors are
/// zero-sized factory tags; the actual per-call state (device
/// handle, persistent const cache) lives in the [`LazyRealizer`]
/// returned by [`BackendFactory::try_make_realizer`].
pub trait BackendFactory: Send + Sync {
    /// Stable identifier — the same one the probe uses.
    fn id(&self) -> BackendId;

    /// Devices this backend currently sees on the host. Wraps each
    /// backend crate's existing `probe::enumerate_devices`.
    fn enumerate_devices(&self) -> Result<Vec<DeviceDescriptor>>;

    /// Construct a fresh realizer pinned to one device. Errors are
    /// propagated as-is — judge.rs prints and skips that device.
    fn try_make_realizer(&self, device_index: u32) -> Result<Box<dyn LazyRealizer>>;
}

/// All backend factories compiled into this build, in the same order
/// the probe used to list them. Plain CPU is always present; the
/// rest gate on cargo features.
pub fn registry() -> Vec<&'static dyn BackendFactory> {
    #[allow(unused_mut)]
    let mut v: Vec<&'static dyn BackendFactory> = vec![
        &CpuFactory,
    ];
    #[cfg(feature = "cuda")]
    v.push(&CudaFactory);
    #[cfg(feature = "vulkan")]
    v.push(&VulkanFactory);
    v
}

/// Look up a factory by its stable [`BackendId`]. `None` if the backend
/// isn't compiled into this build.
pub fn factory_for(id: BackendId) -> Option<&'static dyn BackendFactory> {
    registry().into_iter().find(|f| f.id() == id)
}

// ---------------------------------------------------------------------
// CPU
// ---------------------------------------------------------------------

pub struct CpuFactory;

impl BackendFactory for CpuFactory {
    fn id(&self) -> BackendId { BackendId::Cpu }
    fn enumerate_devices(&self) -> Result<Vec<DeviceDescriptor>> {
        fuel_cpu_backend::probe::enumerate_devices()
    }
    fn try_make_realizer(&self, _device_index: u32) -> Result<Box<dyn LazyRealizer>> {
        Ok(Box::new(BridgeRealizer::new(crate::Device::cpu())))
    }
}

// AOCL + MKL factories retired 2026-06-08 (backend-extensions
// Phase 2). They were vestigial after AOCL/MKL kernels became
// kernel-source extensions of `BackendId::Cpu` via the binding
// table — the factories created realizers identical to
// `CpuFactory`'s. The picker now selects among CPU-substrate
// alternatives by `kernel_source`; no per-vendor factory needed.

// ---------------------------------------------------------------------
// CUDA
// ---------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub struct CudaFactory;

#[cfg(feature = "cuda")]
impl BackendFactory for CudaFactory {
    fn id(&self) -> BackendId { BackendId::Cuda }
    fn enumerate_devices(&self) -> Result<Vec<DeviceDescriptor>> {
        fuel_cuda_backend::probe::enumerate_devices()
    }
    fn try_make_realizer(&self, device_index: u32) -> Result<Box<dyn LazyRealizer>> {
        let dev = fuel_cuda_backend::CudaDevice::new(device_index as usize)
            .map_err(|e| fuel_core_types::Error::Msg(
                format!("CudaDevice::new({device_index}) failed: {e}")
            ))?;
        Ok(Box::new(BridgeRealizer::new(dev.into())))
    }
}

// ---------------------------------------------------------------------
// Vulkan
// ---------------------------------------------------------------------

#[cfg(feature = "vulkan")]
pub struct VulkanFactory;

#[cfg(feature = "vulkan")]
impl BackendFactory for VulkanFactory {
    fn id(&self) -> BackendId { BackendId::Vulkan }
    fn enumerate_devices(&self) -> Result<Vec<DeviceDescriptor>> {
        fuel_vulkan_backend::probe::enumerate_devices()
    }
    fn try_make_realizer(&self, device_index: u32) -> Result<Box<dyn LazyRealizer>> {
        let backend = fuel_vulkan_backend::VulkanBackend::with_selection(
            fuel_vulkan_backend::DeviceSelection::Index(device_index as usize),
        ).map_err(|e| fuel_core_types::Error::Msg(
            format!("VulkanBackend init failed: {e}")
        ))?;
        Ok(Box::new(BridgeRealizer::new(backend.into())))
    }
}
