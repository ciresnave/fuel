//! Stub Metal backend used when the `metal` feature is disabled.
//!
//! These types exist so that `fuel_core::{MetalDevice, MetalStorage,
//! MetalError}` always resolves to *something*. After step 8 of the
//! backend-agnostic refactor (2026-04-30), the static `BackendStorage` /
//! `BackendDevice` traits no longer exist, so this file holds only the stub
//! types.
#![allow(dead_code)]

#[derive(Debug, Clone)]
pub struct MetalDevice;

#[derive(Debug)]
pub struct MetalStorage;

#[derive(thiserror::Error, Debug)]
pub enum MetalError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for MetalError {
    fn from(e: String) -> Self {
        MetalError::Message(e)
    }
}
