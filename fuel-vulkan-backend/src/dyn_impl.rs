//! `DynBackendDevice` implementation for the Vulkan backend.
//!
//! `VulkanBackendDevice` is the wrapper type that satisfies
//! [`DynBackendDevice`]. It carries an `Arc<VulkanBackend>` so a
//! `fuel_core::Device` constructed from this wrapper can be downcast
//! back to the concrete backend at dispatch time (the binding-table
//! dispatch path reaches the backend through `Arc<VulkanBackend>`
//! fields on `VulkanStorageBytes`).
//!
//! All `*_dyn` methods that return `Box<dyn DynBackendStorage>` are
//! stubs that error out. Vulkan storage flows through the byte-shape
//! [`crate::VulkanStorageBytes`] substrate (held inside
//! [`fuel_storage::Storage`]), not the op-rich `DynBackendStorage`
//! trait. Callers use the byte-storage surface on `VulkanBackend`
//! ([`VulkanBackend::alloc_bytes_handle`],
//! [`VulkanBackend::upload_bytes_handle`],
//! [`VulkanBackend::download_bytes`]) reached through
//! `fuel_core::vulkan_backend::as_device(&device)`.
//!
//! [`VulkanBackend::alloc_bytes_handle`]: crate::VulkanBackend::alloc_bytes_handle
//! [`VulkanBackend::upload_bytes_handle`]: crate::VulkanBackend::upload_bytes_handle
//! [`VulkanBackend::download_bytes`]: crate::VulkanBackend::download_bytes

use std::any::Any;
use std::sync::{Arc, Mutex};

use fuel_core_types::dyn_backend::{DynBackendDevice, DynBackendStorage};
use fuel_core_types::{DType, DeviceLocation, Error, HostBuffer, Result, Shape};

use crate::VulkanBackend;

/// `DynBackendDevice` adapter for [`VulkanBackend`].
pub struct VulkanBackendDevice {
    inner: Arc<VulkanBackend>,
    /// RNG seed slot — Vulkan has no native RNG yet, so set/get
    /// merely round-trip a host-side `u64`. Tracked so calls to
    /// [`fuel_core::Device::set_seed`] / `Device::get_current_seed`
    /// don't error out for downstream code that calls them
    /// unconditionally across backends.
    seed: Mutex<u64>,
}

// Manual Debug — `VulkanBackend` does not derive Debug (its vulkane
// fields contain `*mut c_void` pNext pointers that aren't Debug).
// Summarize the useful identifying fields by hand.
impl std::fmt::Debug for VulkanBackendDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBackendDevice")
            .field("gpu_id", &self.inner.gpu_id)
            .field("device_name", &self.inner.device_name)
            .finish()
    }
}

impl VulkanBackendDevice {
    /// Wrap an `Arc<VulkanBackend>` so it can stand in as an
    /// `Arc<dyn DynBackendDevice>` for `fuel_core::Device`.
    pub fn new(inner: Arc<VulkanBackend>) -> Self {
        Self { inner, seed: Mutex::new(0) }
    }

    /// Borrow the wrapped `Arc<VulkanBackend>`.
    ///
    /// `fuel_core::vulkan_backend::as_device` downcasts a `&Device`
    /// to `&VulkanBackendDevice` and clones this `Arc` so the byte-
    /// storage allocation path can call
    /// [`VulkanBackend::alloc_bytes_handle`] etc.
    ///
    /// [`VulkanBackend::alloc_bytes_handle`]: crate::VulkanBackend::alloc_bytes_handle
    pub fn inner(&self) -> &Arc<VulkanBackend> {
        &self.inner
    }
}

fn vulkan_dyn_err(method: &'static str) -> Error {
    Error::Msg(format!(
        "VulkanBackendDevice::{method}: Vulkan does not return DynBackendStorage \
         objects. The Vulkan storage path runs on the byte-shape substrate \
         (VulkanStorageBytes via fuel_storage::Storage). Use the dedicated \
         allocation surface on VulkanBackend (alloc_bytes_handle, \
         upload_bytes_handle, download_bytes) reached via \
         fuel_core::vulkan_backend::as_device(&device).",
    ))
    .bt()
}

impl DynBackendDevice for VulkanBackendDevice {
    fn location_dyn(&self) -> DeviceLocation {
        DeviceLocation::Vulkan { gpu_id: self.inner.gpu_id }
    }

    fn same_device_dyn(&self, other: &dyn DynBackendDevice) -> bool {
        other
            .as_any()
            .downcast_ref::<VulkanBackendDevice>()
            .is_some_and(|o| Arc::ptr_eq(&self.inner, &o.inner))
    }

    fn supports_bf16(&self) -> bool {
        true
    }

    fn zeros_impl_dyn(&self, _shape: &Shape, _dtype: DType) -> Result<Box<dyn DynBackendStorage>> {
        Err(vulkan_dyn_err("zeros_impl_dyn"))
    }

    unsafe fn alloc_uninit_dyn(
        &self,
        _shape: &Shape,
        _dtype: DType,
    ) -> Result<Box<dyn DynBackendStorage>> {
        Err(vulkan_dyn_err("alloc_uninit_dyn"))
    }

    fn storage_from_host_buffer_dyn(
        &self,
        _buf: &HostBuffer,
    ) -> Result<Box<dyn DynBackendStorage>> {
        Err(vulkan_dyn_err("storage_from_host_buffer_dyn"))
    }

    fn storage_from_host_buffer_owned_dyn(
        &self,
        _buf: HostBuffer,
    ) -> Result<Box<dyn DynBackendStorage>> {
        Err(vulkan_dyn_err("storage_from_host_buffer_owned_dyn"))
    }

    fn rand_uniform_dyn(
        &self,
        _shape: &Shape,
        _dtype: DType,
        _lo: f64,
        _hi: f64,
    ) -> Result<Box<dyn DynBackendStorage>> {
        Err(vulkan_dyn_err("rand_uniform_dyn"))
    }

    fn rand_normal_dyn(
        &self,
        _shape: &Shape,
        _dtype: DType,
        _mean: f64,
        _std: f64,
    ) -> Result<Box<dyn DynBackendStorage>> {
        Err(vulkan_dyn_err("rand_normal_dyn"))
    }

    fn set_seed_dyn(&self, seed: u64) -> Result<()> {
        *self.seed.lock().unwrap() = seed;
        Ok(())
    }

    fn get_current_seed_dyn(&self) -> Result<u64> {
        Ok(*self.seed.lock().unwrap())
    }

    fn synchronize_dyn(&self) -> Result<()> {
        self.inner.synchronize_pending()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
