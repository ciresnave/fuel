//! Metal-side `BackendStorage` variant.
//!
//! A1 (this commit): placeholder. Real fields land in A3 (MTLBuffer
//! handle, owning device Arc).

/// **A1 placeholder.** Real fields land in A3.
#[derive(Debug)]
pub struct MetalStorage {
    _placeholder: usize,
}

impl MetalStorage {
    pub fn len_bytes(&self) -> usize {
        self._placeholder
    }
}
