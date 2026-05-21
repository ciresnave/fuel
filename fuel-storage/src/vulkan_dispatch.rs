//! Dispatch wrappers + registration for Vulkan-backed kernels.
//! Sibling to `baracuda_dispatch` (CUDA) and the CPU dispatch in
//! `dispatch.rs`. Each kernel registers as an alternative at the
//! `(op_kind, dtypes, BackendId::Vulkan)` decision point in the
//! shared `KernelBindingTable`.
//!
//! ## V.1 status
//!
//! Phase 7.6 step 9c Vulkan catch-up V.1.C — the proof-of-life
//! milestone. One op (`Add f32`) wired end-to-end through the
//! pipelined-executor binding-table dispatch:
//! 1. The executor's [`pipelined::execute_work_item`] output-
//!    allocation match arm sees `BackendId::Vulkan` and allocates
//!    via [`fuel_vulkan_backend::VulkanBackend::alloc_bytes_handle`]
//!    (V.1.B).
//! 2. The kernel wrapper here reads inputs (also `VulkanStorageBytes`
//!    with backend handle attached), pulls the backend Arc from the
//!    first input, and calls
//!    [`fuel_vulkan_backend::VulkanBackend::binary_add_f32_bytes`]
//!    to dispatch.
//! 3. The Slang `binary.spv` kernel runs on the Vulkan device; the
//!    pre-allocated output buffer carries the result.
//!
//! V.2 fans out registrations to the ~25 already-compiled SPIR-V
//! kernels in `fuel-vulkan-kernels`. V.3 writes new Slang for the
//! kernel families CUDA has and Vulkan doesn't.

#![cfg(feature = "vulkan")]

use std::sync::{Arc, RwLock};

use fuel_core_types::dispatch::OpKind;
use fuel_core_types::probe::BackendId;
use fuel_core_types::{DType, Error, Layout, Result};

use crate::kernel::{KernelBindingTable, OpParams};
use crate::Storage;

// Re-use the storage-lock helpers from dispatch.rs.
use crate::dispatch::{read_storage, write_storage};

/// Helper: extract `&VulkanStorageBytes` from `&Storage`. Returns
/// `Err` if the variant isn't `BackendStorage::Vulkan`.
fn vulkan_input(s: &Storage) -> Result<&fuel_vulkan_backend::VulkanStorageBytes> {
    match &s.inner {
        crate::BackendStorage::Vulkan(v) => Ok(v),
        #[allow(unreachable_patterns)]
        _ => Err(Error::Msg(
            "vulkan kernel wrapper called with non-Vulkan input".to_string(),
        )
        .bt()),
    }
}

/// Helper: extract `&mut VulkanStorageBytes` from `&mut Storage`.
fn vulkan_output(s: &mut Storage) -> Result<&mut fuel_vulkan_backend::VulkanStorageBytes> {
    match &mut s.inner {
        crate::BackendStorage::Vulkan(v) => Ok(v),
        #[allow(unreachable_patterns)]
        _ => Err(Error::Msg(
            "vulkan kernel wrapper called with non-Vulkan output".to_string(),
        )
        .bt()),
    }
}

// ===========================================================================
// Binary — element-wise (V.1.C proof-of-life: Add f32 only)
// ===========================================================================
//
// Mirrors baracuda_dispatch's `binary::add_f32` shape. Takes 2
// inputs + 1 pre-allocated output. V.2 fans out to Sub/Mul/Div +
// other dtypes via per-op-id selection in the `binary.slang` kernel
// (which has a `op_id` push constant covering Add/Sub/Mul/Div).

pub mod binary {
    use super::*;

    pub fn add_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        _params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::binary::add_f32: expected 2 inputs + 1 output, \
                 got {} + {}",
                inputs.len(),
                outputs.len(),
            ))
            .bt());
        }
        if layouts.len() < 2 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::binary::add_f32: layouts.len() = {} < 2 (need \
                 per-input strides for broadcast-binary dispatch)",
                layouts.len(),
            ))
            .bt());
        }
        let in0_guard = read_storage(&inputs[0])?;
        let in1_guard = read_storage(&inputs[1])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let a = vulkan_input(&in0_guard)?;
        let b = vulkan_input(&in1_guard)?;

        // Pull the backend handle off input[0]. The pipelined
        // executor's output-allocation arm (V.1.B) guarantees both
        // the inputs and the output were allocated via
        // alloc_bytes_handle so the Arc is present; if a caller
        // bypassed that path the handle is None and we error
        // cleanly.
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::binary::add_f32: input[0] has no \
                 VulkanBackend handle. Storages flowing through the \
                 pipelined-executor binding-table dispatch must come \
                 from alloc_bytes_handle / upload_bytes_handle."
                    .to_string(),
            )
            .bt()
        })?;

        let out = vulkan_output(&mut out_guard)?;
        backend.binary_add_f32_bytes(a, b, out, &layouts[0], &layouts[1])
    }
}

// ===========================================================================
// register_vulkan_kernels — binding-table population
// ===========================================================================

/// Register every Vulkan kernel wrapper against its `(OpKind, dtypes,
/// BackendId::Vulkan)` decision-point key in the shared
/// `KernelBindingTable`. Each wrapper appears as an alternative
/// alongside any CPU / CUDA / future-backend registrations at the
/// same key. The route picker (architecture v1.0 §04) selects among
/// them at plan time based on input residency + telemetry.
///
/// V.1.C coverage: `AddElementwise [F32, F32, F32]` only. The
/// proof-of-life single op. V.2 fans out to the ~25 already-compiled
/// SPIR-V kernels; V.3 adds new Slang for the kernel families CUDA
/// has and Vulkan doesn't.
pub fn register_vulkan_kernels(table: &mut KernelBindingTable) {
    let vk = BackendId::Vulkan;
    let f32 = DType::F32;

    // ----- Binary (Add f32 only — V.1.C proof-of-life) -----
    table.register(
        OpKind::AddElementwise,
        &[f32, f32, f32],
        vk,
        binary::add_f32,
    );
}
