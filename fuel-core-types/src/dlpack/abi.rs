//! Standard DLPack v1.x ABI — reproduced verbatim from `dlpack.h` (v1.3).
//!
//! FDX (the Fuel DLPack eXtension) *consumes these unchanged*; it never
//! redefines them. The base `DLTensor` is always honest, conformant standard
//! DLPack on its own — see `docs/specs/dlpack-extension.md` §3, §5.1.
//!
//! These are `#[repr(C)]` POD with a layout that matches the C header
//! byte-for-byte on a 64-bit little-endian host (the v1 target; §5).

use core::ffi::c_void;

/// DLPack device-type codes (`dlpack.h` `DLDeviceType`); the subset FDX uses.
pub mod device_type {
    pub const K_DL_CPU: i32 = 1;
    pub const K_DL_CUDA: i32 = 2;
    pub const K_DL_CUDA_HOST: i32 = 3;
    pub const K_DL_VULKAN: i32 = 7;
    pub const K_DL_METAL: i32 = 8;
}

/// DLPack dtype-code field of [`DLDataType`] (`dlpack.h` `DLDataTypeCode`).
pub mod dtype_code {
    pub const K_DL_INT: u8 = 0;
    pub const K_DL_UINT: u8 = 1;
    pub const K_DL_FLOAT: u8 = 2;
    pub const K_DL_BFLOAT: u8 = 4;
    pub const K_DL_COMPLEX: u8 = 5;
    pub const K_DL_BOOL: u8 = 6;
}

/// Standard DLPack flags carried on [`DLManagedTensorVersioned::flags`]
/// (`dlpack.h`). FDX relies on these directly and never redefines them.
pub const DLPACK_FLAG_BITMASK_READ_ONLY: u64 = 1 << 0;
/// Set by a producer that materialized a copy (dequantize / live-prefix /
/// aligned copy) — see spec §9.1.
pub const DLPACK_FLAG_BITMASK_IS_COPIED: u64 = 1 << 1;
/// Sub-byte packed/padded marker. FDX never sets it (§3.4 — sub-byte payloads
/// ride the `uint8` honesty stand-in, not the native DLPack sub-byte path).
pub const DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED: u64 = 1 << 2;

/// `dlpack.h` `DLDevice`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DLDevice {
    pub device_type: i32,
    pub device_id: i32,
}

/// `dlpack.h` `DLDataType` — `code`/`bits`/`lanes`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

/// `dlpack.h` `DLTensor`. `data` is 256-byte aligned on export (§3.3); the
/// logical start rides `byte_offset`. `strides` is length `ndim`, **never
/// NULL** on a versioned export (§3.2), and may be negative — first-class in
/// FDX (§3.2.1).
#[repr(C)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    /// length `ndim`; capacity bounds for symbolic axes.
    pub shape: *mut i64,
    /// length `ndim`, never NULL on a versioned export; keyed to capacity.
    pub strides: *mut i64,
    pub byte_offset: u64,
}

/// `dlpack.h` `DLPackVersion`. The DLPack **ABI** version — independent of the
/// FDX schema version (§5.2).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DLPackVersion {
    pub major: u32,
    pub minor: u32,
}

/// `dlpack.h` `DLManagedTensorVersioned` — the cross-runtime managed form.
/// At boundary (b) the FDX sidecar rides `manager_ctx`, recovered only when
/// the live `deleter` identity matches Fuel's own (§10.2). The deleter and
/// `manager_ctx` pointers are never serialized.
#[repr(C)]
pub struct DLManagedTensorVersioned {
    pub version: DLPackVersion,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    pub flags: u64,
    pub dl_tensor: DLTensor,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    // ABI byte-for-byte size pins (64-bit LE target, §5). A struct-layout
    // change that would break the C-header match fails the build here.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn dlpack_struct_sizes_match_c_header() {
        assert_eq!(size_of::<DLDevice>(), 8);
        assert_eq!(size_of::<DLDataType>(), 4);
        assert_eq!(size_of::<DLPackVersion>(), 8);
        assert_eq!(size_of::<DLTensor>(), 48);
        assert_eq!(size_of::<DLManagedTensorVersioned>(), 80);
    }

    #[test]
    fn dlpack_standard_flag_bits() {
        assert_eq!(DLPACK_FLAG_BITMASK_READ_ONLY, 1);
        assert_eq!(DLPACK_FLAG_BITMASK_IS_COPIED, 2);
        assert_eq!(DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED, 4);
    }
}
