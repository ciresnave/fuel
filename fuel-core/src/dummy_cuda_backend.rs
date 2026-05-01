//! Stub Cuda backend used when the `cuda` feature is disabled.
//!
//! These types exist so that `fuel_core::{CudaDevice, CudaStorage}` always
//! resolves to *something*; every method panics or returns
//! `Error::NotCompiledWithCudaSupport`. After step 8 of the backend-agnostic
//! refactor (2026-04-30), the static `BackendStorage` / `BackendDevice`
//! traits no longer exist, so this file holds only the stub types and a
//! couple of placeholder constructors.
#![allow(dead_code)]

use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct CudaDevice;

#[derive(Debug)]
pub struct CudaStorage;

impl CudaStorage {
    pub fn transfer_to_device(&self, _dst: &CudaDevice) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl CudaDevice {
    pub fn new_with_stream(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }
    pub fn id(&self) -> DeviceId {
        DeviceId(0)
    }
}
