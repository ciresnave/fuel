//! Re-exports of CPU storage types and helpers.
//!
//! After step 8 of the backend-agnostic refactor (2026-04-30), all real CPU
//! kernel logic lives in `fuel-cpu-backend`. This module exists only to keep
//! the crate path `fuel_core::cpu_backend::*` stable for downstream
//! consumers (notably `fuel-nn` which calls `unary_map` here).

pub use fuel_ir::{CpuDevice, CpuStorage, CpuStorageRef, HostBuffer, HostBufferRef};

pub use fuel_cpu_backend::utils::{
    binary_map, binary_map_vec, unary_map, unary_map_vec,
    Map1, Map1Any, Map2, Map2InPlace, Map2U8,
};
