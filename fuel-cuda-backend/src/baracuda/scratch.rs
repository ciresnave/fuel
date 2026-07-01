//! Per-call workspace allocation for baracuda kernels.
//!
//! Each baracuda kernel has a `_workspace_size(...)` query that
//! reports its required scratch in bytes. The launch site allocates a
//! fresh `DeviceBuffer<u8>` of that size, passes its pointer + length,
//! and drops it when the kernel returns.
//!
//! ## Pooling (deferred)
//!
//! A per-stream scratch pool — reuse one large device buffer across
//! kernel launches — is an obvious optimization but not yet
//! implemented. Today's alloc-per-call model is correct, the typical
//! transformer launch has O(layers × heads) kernel invocations which
//! is bounded, and `cuMemAlloc` is fast enough on modern drivers that
//! this hasn't shown up as a measurable hotspot. When it does, the
//! pool lives here.

use baracuda_driver::DeviceBuffer;
use fuel_ir::Result;

use crate::CudaDevice;

/// Workspace buffer for one kernel launch.
///
/// Holds the underlying `DeviceBuffer<u8>` so it stays live for the
/// duration of the launch (the kernel writes into it as scratch).
/// Drop frees the device memory; no manual cleanup needed.
pub struct Workspace {
    buf: Option<DeviceBuffer<u8>>,
    bytes: usize,
}

impl Workspace {
    /// Allocate a fresh workspace of `bytes` bytes on `device`. When
    /// `bytes == 0` returns a no-op workspace whose `as_raw` is a
    /// null pointer — matches baracuda's "no scratch needed" contract.
    pub fn alloc(device: &CudaDevice, bytes: usize) -> Result<Self> {
        if bytes == 0 {
            return Ok(Self { buf: None, bytes: 0 });
        }
        let buf = device.alloc_zeros::<u8>(bytes)?;
        Ok(Self {
            buf: Some(buf),
            bytes,
        })
    }

    /// Raw device pointer for the kernel-launch ABI. `null` when
    /// `bytes == 0`.
    pub fn as_raw(&self) -> *mut std::ffi::c_void {
        match self.buf.as_ref() {
            Some(b) => b.as_raw().0 as *mut std::ffi::c_void,
            None => std::ptr::null_mut(),
        }
    }

    /// Byte size — what the kernel sees as `workspace_bytes`.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}
