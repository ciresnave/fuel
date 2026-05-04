//! CUDA-side `BackendStorage` variant.
//!
//! A1 (this commit): placeholder. Real fields land in A3 (CUdeviceptr,
//! length, owning context Arc).

/// **A1 placeholder.** Real fields land in A3.
#[derive(Debug)]
pub struct CudaStorage {
    /// Placeholder so the type isn't a ZST; A3 replaces with real
    /// CUDA handle + context.
    _placeholder: usize,
}

impl CudaStorage {
    /// Total byte count. A1 always reports 0 since there's no real
    /// storage yet.
    pub fn len_bytes(&self) -> usize {
        self._placeholder
    }
}
