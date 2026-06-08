//! Backend factory registry — Step 12 of the backend-agnostic refactor.
//!
//! Every backend that fuel-core can drive at runtime declares a single
//! [`BackendFactory`] value in this module. The registry ([`registry`])
//! returns the cfg-gated subset that's actually compiled in, and
//! consumers — currently the [`crate::probe`] enumerator and the
//! [`crate::judge`] profiler — walk that registry instead of naming
//! `fuel_cuda_backend::CudaBackend`/`fuel_aocl_cpu_backend::AoclBackend`/...
//! by hand.
//!
//! # Why a fuel-core-local trait
//!
//! [`LazyRealizer`] returns a `Vec<f32>` from a [`crate::lazy::LazyTensor`].
//! Both types are owned by fuel-core, so the trait can't sit in
//! fuel-core-types without dragging LazyTensor down with it. Each
//! factory impl below therefore lives in fuel-core (one cfg-gated block
//! per backend) and constructs the typed `GraphExecutor<B>` internally,
//! returning a `Box<dyn LazyRealizer>` that owns it. judge.rs ends up
//! with zero references to specific backend types.
//!
//! # Adding a new backend
//!
//! 1. Add a unit struct `MyBackendFactory` here, behind a cfg(feature).
//! 2. Implement `BackendFactory` — `enumerate_devices` delegates to the
//!    backend crate's existing `probe::enumerate_devices`, and
//!    `try_make_realizer` constructs the typed executor and wraps it
//!    in the local `Realizer<B>` adapter.
//! 3. Add a cfg-gated entry in [`registry`].
//!
//! No edits to judge.rs or probe.rs.

use crate::lazy::LazyTensor;
use fuel_core_types::probe::{BackendId, DeviceDescriptor};
use fuel_core_types::Result;
use fuel_graph_executor::{GraphBackend, GraphExecutor};

/// Object-safe wrapper around a typed `GraphExecutor<B>` that exposes
/// just the f32 realize entry point used by judge.rs. Each factory
/// returns a `Box<dyn LazyRealizer>`, hiding the concrete `B` from
/// the caller.
pub trait LazyRealizer {
    fn realize_f32(&mut self, tensor: &LazyTensor) -> Vec<f32>;
}

/// Generic adapter: any `GraphExecutor<B>` becomes a `LazyRealizer`.
struct Realizer<B: GraphBackend> {
    exe: GraphExecutor<B>,
}

impl<B: GraphBackend> LazyRealizer for Realizer<B> {
    fn realize_f32(&mut self, tensor: &LazyTensor) -> Vec<f32> {
        self.exe.realize_f32(tensor.graph_tensor()).into_vec()
    }
}

/// One concrete backend the runtime can drive. Implementors are
/// zero-sized factory tags; the actual per-call state (typed executor,
/// CUDA context, etc.) lives in the [`LazyRealizer`] returned by
/// [`try_make_realizer`].
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
/// the probe used to list them. Reference + plain CPU are always
/// present; the rest gate on cargo features.
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
// CPU (fuel-graph-cpu)
// ---------------------------------------------------------------------

pub struct CpuFactory;

impl BackendFactory for CpuFactory {
    fn id(&self) -> BackendId { BackendId::Cpu }
    fn enumerate_devices(&self) -> Result<Vec<DeviceDescriptor>> {
        fuel_cpu_backend::probe::enumerate_devices()
    }
    fn try_make_realizer(&self, _device_index: u32) -> Result<Box<dyn LazyRealizer>> {
        Ok(Box::new(Realizer {
            exe: GraphExecutor::new(fuel_graph_cpu::CpuBackend),
        }))
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
        let backend = fuel_cuda_backend::CudaBackend::new(dev);
        Ok(Box::new(Realizer { exe: GraphExecutor::new(backend) }))
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
        Ok(Box::new(Realizer { exe: GraphExecutor::new(backend) }))
    }
}
