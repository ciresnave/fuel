//! Vulkan-side `BackendStorage` variant.
//!
//! A1 (this commit): placeholder. Real fields land in A3 (VkBuffer,
//! VmaAllocation, owning device Arc).

/// **A1 placeholder.** Real fields land in A3.
#[derive(Debug)]
pub struct VulkanStorage {
    _placeholder: usize,
}

impl VulkanStorage {
    pub fn len_bytes(&self) -> usize {
        self._placeholder
    }
}
