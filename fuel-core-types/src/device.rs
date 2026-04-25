//! Identifies a physical device location (CPU or a specific GPU).

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Identifies a physical device location (CPU or a specific GPU).
///
/// # Example
///
/// ```rust
/// use fuel_core_types::DeviceLocation;
/// let loc = DeviceLocation::Cpu;
/// assert_eq!(loc, DeviceLocation::Cpu);
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DeviceLocation {
    /// The CPU.
    Cpu,
    /// A CUDA GPU with the given ordinal.
    Cuda { gpu_id: usize },
    /// A Metal GPU with the given ordinal.
    Metal { gpu_id: usize },
    /// A Vulkan GPU with the given ordinal.
    Vulkan { gpu_id: usize },
}
