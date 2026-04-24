//! Alternative [`HostStorage`](fuel_core_types::backend::HostStorage) impls
//! beyond the default owned [`CpuBackendStorage`](crate::dyn_impl::
//! CpuBackendStorage).
//!
//! Each sub-module adds a new source of host-resident bytes that integrates
//! into Fuel's upload path via the `HostStorage` trait. Consumers pick the
//! one matching their data source:
//!
//! - [`mmap::MmappedHostStorage`] — memory-mapped file, for zero-copy
//!   safetensors / GGUF loading.
//! - (future) `PinnedHostStorage` — page-locked RAM for GPU DMA.
//! - (future) `SharedMemHostStorage` — cross-process shared regions.

pub mod mmap;
pub mod shared_mem;

pub use mmap::MmappedHostStorage;
pub use shared_mem::SharedMemHostStorage;
