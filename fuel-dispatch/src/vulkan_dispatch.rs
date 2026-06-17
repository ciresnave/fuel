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
use fuel_memory::{BackendStorage, Storage};

// Re-use the storage-lock helpers from dispatch.rs.
use crate::dispatch::{read_storage, write_storage};

/// Helper: extract `&VulkanStorageBytes` from `&Storage`. Returns
/// `Err` if the variant isn't `BackendStorage::Vulkan`.
fn vulkan_input(s: &Storage) -> Result<&fuel_vulkan_backend::VulkanStorageBytes> {
    match &s.inner {
        fuel_memory::BackendStorage::Vulkan(v) => Ok(v),
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
        fuel_memory::BackendStorage::Vulkan(v) => Ok(v),
        #[allow(unreachable_patterns)]
        _ => Err(Error::Msg(
            "vulkan kernel wrapper called with non-Vulkan output".to_string(),
        )
        .bt()),
    }
}

// ===========================================================================
// Binary — element-wise f32 (V.2.A: Add/Sub/Mul/Div/Max/Min)
// ===========================================================================
//
// One wrapper per op; all six route through
// `VulkanBackend::binary_f32_bytes(op_id, ...)` which uses the same
// `binary.spv` Slang kernel with the `op_id` push constant
// selecting the actual math. Mirrors baracuda_dispatch's per-op
// wrapper shape.

/// Generate one binary-f32 KernelRef-shaped wrapper that calls
/// `VulkanBackend::binary_f32_bytes` with the given op_id + label.
macro_rules! vk_binary_f32_wrapper {
    ($name:ident, $op_id:expr, $label:expr $(,)?) => {
        pub fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            layouts: &[Layout],
            _params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::binary::{}: expected 2 inputs + 1 output, got {} + {}",
                    $label, inputs.len(), outputs.len(),
                )).bt());
            }
            if layouts.len() < 2 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::binary::{}: layouts.len() = {} < 2",
                    $label, layouts.len(),
                )).bt());
            }
            let in0_guard = read_storage(&inputs[0])?;
            let in1_guard = read_storage(&inputs[1])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let a = vulkan_input(&in0_guard)?;
            let b = vulkan_input(&in1_guard)?;
            let backend = a.backend().ok_or_else(|| {
                Error::Msg(format!(
                    "vulkan_dispatch::binary::{}: input[0] has no VulkanBackend handle. \
                     Storages flowing through the pipelined-executor binding-table \
                     dispatch must come from alloc_bytes_handle / upload_bytes_handle.",
                    $label,
                )).bt()
            })?;
            let out = vulkan_output(&mut out_guard)?;
            backend.binary_f32_bytes($op_id, $label, a, b, out, &layouts[0], &layouts[1])
        }
    };
}

pub mod binary {
    use super::*;

    vk_binary_f32_wrapper!(add_f32, 0, "binary_add_f32");
    vk_binary_f32_wrapper!(sub_f32, 1, "binary_sub_f32");
    vk_binary_f32_wrapper!(mul_f32, 2, "binary_mul_f32");
    vk_binary_f32_wrapper!(div_f32, 3, "binary_div_f32");
    vk_binary_f32_wrapper!(maximum_f32, 4, "binary_maximum_f32");
    vk_binary_f32_wrapper!(minimum_f32, 5, "binary_minimum_f32");
}

// V.3.E — f16 binary fan-out via native float16_t.

macro_rules! vk_binary_f16_wrapper {
    ($name:ident, $op_id:expr, $label:expr $(,)?) => {
        pub fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            layouts: &[Layout],
            _params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::binary_f16::{}: expected 2 inputs + 1 output, got {} + {}",
                    $label, inputs.len(), outputs.len(),
                )).bt());
            }
            if layouts.len() < 2 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::binary_f16::{}: layouts.len() = {} < 2",
                    $label, layouts.len(),
                )).bt());
            }
            let in0_guard = read_storage(&inputs[0])?;
            let in1_guard = read_storage(&inputs[1])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let a = vulkan_input(&in0_guard)?;
            let b = vulkan_input(&in1_guard)?;
            let backend = a.backend().ok_or_else(|| {
                Error::Msg(format!(
                    "vulkan_dispatch::binary_f16::{}: input[0] has no VulkanBackend handle. \
                     Storages flowing through the pipelined-executor binding-table \
                     dispatch must come from alloc_bytes_handle / upload_bytes_handle.",
                    $label,
                )).bt()
            })?;
            let out = vulkan_output(&mut out_guard)?;
            backend.binary_f16_bytes($op_id, $label, a, b, out, &layouts[0], &layouts[1])
        }
    };
}

pub mod binary_f16 {
    use super::*;

    vk_binary_f16_wrapper!(add_f16,     0, "binary_add_f16");
    vk_binary_f16_wrapper!(sub_f16,     1, "binary_sub_f16");
    vk_binary_f16_wrapper!(mul_f16,     2, "binary_mul_f16");
    vk_binary_f16_wrapper!(div_f16,     3, "binary_div_f16");
    vk_binary_f16_wrapper!(maximum_f16, 4, "binary_maximum_f16");
    vk_binary_f16_wrapper!(minimum_f16, 5, "binary_minimum_f16");
}

// V.3.E.5 — f64 binary fan-out via native `double`.

macro_rules! vk_binary_f64_wrapper {
    ($name:ident, $op_id:expr, $label:expr $(,)?) => {
        pub fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            layouts: &[Layout],
            _params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::binary_f64::{}: expected 2 inputs + 1 output, got {} + {}",
                    $label, inputs.len(), outputs.len(),
                )).bt());
            }
            if layouts.len() < 2 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::binary_f64::{}: layouts.len() = {} < 2",
                    $label, layouts.len(),
                )).bt());
            }
            let in0_guard = read_storage(&inputs[0])?;
            let in1_guard = read_storage(&inputs[1])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let a = vulkan_input(&in0_guard)?;
            let b = vulkan_input(&in1_guard)?;
            let backend = a.backend().ok_or_else(|| {
                Error::Msg(format!(
                    "vulkan_dispatch::binary_f64::{}: input[0] has no VulkanBackend handle.",
                    $label,
                )).bt()
            })?;
            let out = vulkan_output(&mut out_guard)?;
            backend.binary_f64_bytes($op_id, $label, a, b, out, &layouts[0], &layouts[1])
        }
    };
}

pub mod binary_f64 {
    use super::*;

    vk_binary_f64_wrapper!(add_f64,     0, "binary_add_f64");
    vk_binary_f64_wrapper!(sub_f64,     1, "binary_sub_f64");
    vk_binary_f64_wrapper!(mul_f64,     2, "binary_mul_f64");
    vk_binary_f64_wrapper!(div_f64,     3, "binary_div_f64");
    vk_binary_f64_wrapper!(maximum_f64, 4, "binary_maximum_f64");
    vk_binary_f64_wrapper!(minimum_f64, 5, "binary_minimum_f64");
}

// ===========================================================================
// Unary — element-wise f32 (V.2.B: 13 ops via unary.slang's op_id)
// ===========================================================================

macro_rules! vk_unary_f32_wrapper {
    ($name:ident, $op_id:expr, $label:expr $(,)?) => {
        pub fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            layouts: &[Layout],
            _params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::unary::{}: expected 1 input + 1 output, got {} + {}",
                    $label, inputs.len(), outputs.len(),
                )).bt());
            }
            if layouts.is_empty() {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::unary::{}: layouts.len() = 0, expected >= 1",
                    $label,
                )).bt());
            }
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let a = vulkan_input(&in_guard)?;
            let backend = a.backend().ok_or_else(|| {
                Error::Msg(format!(
                    "vulkan_dispatch::unary::{}: input has no VulkanBackend handle. \
                     Storages flowing through the pipelined-executor binding-table \
                     dispatch must come from alloc_bytes_handle / upload_bytes_handle.",
                    $label,
                )).bt()
            })?;
            let out = vulkan_output(&mut out_guard)?;
            backend.unary_f32_bytes($op_id, $label, a, out, &layouts[0])
        }
    };
}

pub mod unary {
    use super::*;

    vk_unary_f32_wrapper!(neg_f32,     0,  "unary_neg_f32");
    vk_unary_f32_wrapper!(sqr_f32,     1,  "unary_sqr_f32");
    vk_unary_f32_wrapper!(sqrt_f32,    2,  "unary_sqrt_f32");
    vk_unary_f32_wrapper!(exp_f32,     3,  "unary_exp_f32");
    vk_unary_f32_wrapper!(log_f32,     4,  "unary_log_f32");
    vk_unary_f32_wrapper!(sin_f32,     5,  "unary_sin_f32");
    vk_unary_f32_wrapper!(cos_f32,     6,  "unary_cos_f32");
    vk_unary_f32_wrapper!(tanh_f32,    7,  "unary_tanh_f32");
    vk_unary_f32_wrapper!(sigmoid_f32, 8,  "unary_sigmoid_f32");
    vk_unary_f32_wrapper!(silu_f32,    9,  "unary_silu_f32");
    vk_unary_f32_wrapper!(gelu_f32,    10, "unary_gelu_f32");
    vk_unary_f32_wrapper!(relu_f32,    11, "unary_relu_f32");
    vk_unary_f32_wrapper!(step_f32,    12, "unary_step_f32");
    vk_unary_f32_wrapper!(abs_f32,     13, "unary_abs_f32");
    vk_unary_f32_wrapper!(sign_f32,    14, "unary_sign_f32");
    vk_unary_f32_wrapper!(recip_f32,   15, "unary_recip_f32");
}

// V.3.E — f16 fan-out via native float16_t (shaderFloat16 + 16BitStorage).

macro_rules! vk_unary_f16_wrapper {
    ($name:ident, $op_id:expr, $label:expr $(,)?) => {
        pub fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            layouts: &[Layout],
            _params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::unary_f16::{}: expected 1 input + 1 output, got {} + {}",
                    $label, inputs.len(), outputs.len(),
                )).bt());
            }
            if layouts.is_empty() {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::unary_f16::{}: layouts.len() = 0, expected >= 1",
                    $label,
                )).bt());
            }
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let a = vulkan_input(&in_guard)?;
            let backend = a.backend().ok_or_else(|| {
                Error::Msg(format!(
                    "vulkan_dispatch::unary_f16::{}: input has no VulkanBackend handle. \
                     Storages flowing through the pipelined-executor binding-table \
                     dispatch must come from alloc_bytes_handle / upload_bytes_handle.",
                    $label,
                )).bt()
            })?;
            let out = vulkan_output(&mut out_guard)?;
            backend.unary_f16_bytes($op_id, $label, a, out, &layouts[0])
        }
    };
}

pub mod unary_f16 {
    use super::*;

    vk_unary_f16_wrapper!(neg_f16,     0,  "unary_neg_f16");
    vk_unary_f16_wrapper!(sqr_f16,     1,  "unary_sqr_f16");
    vk_unary_f16_wrapper!(sqrt_f16,    2,  "unary_sqrt_f16");
    vk_unary_f16_wrapper!(exp_f16,     3,  "unary_exp_f16");
    vk_unary_f16_wrapper!(log_f16,     4,  "unary_log_f16");
    vk_unary_f16_wrapper!(sin_f16,     5,  "unary_sin_f16");
    vk_unary_f16_wrapper!(cos_f16,     6,  "unary_cos_f16");
    vk_unary_f16_wrapper!(tanh_f16,    7,  "unary_tanh_f16");
    vk_unary_f16_wrapper!(sigmoid_f16, 8,  "unary_sigmoid_f16");
    vk_unary_f16_wrapper!(silu_f16,    9,  "unary_silu_f16");
    vk_unary_f16_wrapper!(gelu_f16,    10, "unary_gelu_f16");
    vk_unary_f16_wrapper!(relu_f16,    11, "unary_relu_f16");
    vk_unary_f16_wrapper!(step_f16,    12, "unary_step_f16");
    vk_unary_f16_wrapper!(abs_f16,     13, "unary_abs_f16");
    vk_unary_f16_wrapper!(sign_f16,    14, "unary_sign_f16");
    vk_unary_f16_wrapper!(recip_f16,   15, "unary_recip_f16");
}

// V.3.E.5 — f64 fan-out via native `double` (shaderFloat64).

macro_rules! vk_unary_f64_wrapper {
    ($name:ident, $op_id:expr, $label:expr $(,)?) => {
        pub fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            layouts: &[Layout],
            _params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::unary_f64::{}: expected 1 input + 1 output, got {} + {}",
                    $label, inputs.len(), outputs.len(),
                )).bt());
            }
            if layouts.is_empty() {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::unary_f64::{}: layouts.len() = 0, expected >= 1",
                    $label,
                )).bt());
            }
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let a = vulkan_input(&in_guard)?;
            let backend = a.backend().ok_or_else(|| {
                Error::Msg(format!(
                    "vulkan_dispatch::unary_f64::{}: input has no VulkanBackend handle. \
                     Storages flowing through the pipelined-executor binding-table \
                     dispatch must come from alloc_bytes_handle / upload_bytes_handle.",
                    $label,
                )).bt()
            })?;
            let out = vulkan_output(&mut out_guard)?;
            backend.unary_f64_bytes($op_id, $label, a, out, &layouts[0])
        }
    };
}

pub mod unary_f64 {
    use super::*;

    vk_unary_f64_wrapper!(neg_f64,     0,  "unary_neg_f64");
    vk_unary_f64_wrapper!(sqr_f64,     1,  "unary_sqr_f64");
    vk_unary_f64_wrapper!(sqrt_f64,    2,  "unary_sqrt_f64");
    vk_unary_f64_wrapper!(exp_f64,     3,  "unary_exp_f64");
    vk_unary_f64_wrapper!(log_f64,     4,  "unary_log_f64");
    vk_unary_f64_wrapper!(sin_f64,     5,  "unary_sin_f64");
    vk_unary_f64_wrapper!(cos_f64,     6,  "unary_cos_f64");
    vk_unary_f64_wrapper!(tanh_f64,    7,  "unary_tanh_f64");
    vk_unary_f64_wrapper!(sigmoid_f64, 8,  "unary_sigmoid_f64");
    vk_unary_f64_wrapper!(silu_f64,    9,  "unary_silu_f64");
    vk_unary_f64_wrapper!(gelu_f64,    10, "unary_gelu_f64");
    vk_unary_f64_wrapper!(relu_f64,    11, "unary_relu_f64");
    vk_unary_f64_wrapper!(step_f64,    12, "unary_step_f64");
    vk_unary_f64_wrapper!(abs_f64,     13, "unary_abs_f64");
    vk_unary_f64_wrapper!(sign_f64,    14, "unary_sign_f64");
    vk_unary_f64_wrapper!(recip_f64,   15, "unary_recip_f64");
}

// ===========================================================================
// Softmax — last-dim f32 (V.2.C)
// ===========================================================================
//
// One wrapper for Softmax (LogSoftmax is V.3 — no Slang kernel yet).
// Pulls `(outer_count, last_dim)` from `OpParams::SoftmaxLastDim` and
// forwards to `VulkanBackend::softmax_last_dim_f32_bytes`.

pub mod softmax {
    use super::*;

    // Internal helper: extract (outer_count, last_dim) from
    // OpParams::SoftmaxLastDim, with a tagged error on mismatch.
    fn softmax_params(fn_name: &str, params: &OpParams) -> Result<(usize, usize)> {
        match params {
            OpParams::SoftmaxLastDim { outer_count, last_dim } => Ok((*outer_count, *last_dim)),
            other => Err(Error::Msg(format!(
                "vulkan_dispatch::softmax::{fn_name}: expected OpParams::SoftmaxLastDim, got {other:?}",
            )).bt()),
        }
    }

    pub fn softmax_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::softmax::softmax_f32: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (outer_count, last_dim) = softmax_params("softmax_f32", params)?;
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let a = vulkan_input(&in_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::softmax::softmax_f32: input has no VulkanBackend handle. \
                 Storages flowing through the pipelined-executor binding-table dispatch \
                 must come from alloc_bytes_handle / upload_bytes_handle."
                    .to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.softmax_last_dim_f32_bytes(a, out, outer_count, last_dim)
    }

    pub fn softmax_f16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::softmax::softmax_f16: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (outer_count, last_dim) = softmax_params("softmax_f16", params)?;
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let a = vulkan_input(&in_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::softmax::softmax_f16: input has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.softmax_last_dim_f16_bytes(a, out, outer_count, last_dim)
    }

    pub fn softmax_bf16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::softmax::softmax_bf16: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (outer_count, last_dim) = softmax_params("softmax_bf16", params)?;
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let a = vulkan_input(&in_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::softmax::softmax_bf16: input has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.softmax_last_dim_bf16_bytes(a, out, outer_count, last_dim)
    }

    pub fn softmax_f64(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::softmax::softmax_f64: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (outer_count, last_dim) = softmax_params("softmax_f64", params)?;
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let a = vulkan_input(&in_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::softmax::softmax_f64: input has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.softmax_last_dim_f64_bytes(a, out, outer_count, last_dim)
    }

    // ----- SoftmaxLastDimBackward (V.3.G.softmax-bwd, 2026-05-30) -----
    // 2 inputs (y, g) → 1 output (dx). Reuses OpParams::SoftmaxLastDim
    // (same outer × last_dim contract as the forward).

    pub fn softmax_last_dim_backward_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        softmax_backward_typed("softmax_last_dim_backward_f32", inputs, outputs, layouts, params, |b, y, g, dx, oc, ld| {
            b.softmax_last_dim_backward_f32_bytes(y, g, dx, oc, ld)
        })
    }

    pub fn softmax_last_dim_backward_f16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        softmax_backward_typed("softmax_last_dim_backward_f16", inputs, outputs, layouts, params, |b, y, g, dx, oc, ld| {
            b.softmax_last_dim_backward_f16_bytes(y, g, dx, oc, ld)
        })
    }

    pub fn softmax_last_dim_backward_bf16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        softmax_backward_typed("softmax_last_dim_backward_bf16", inputs, outputs, layouts, params, |b, y, g, dx, oc, ld| {
            b.softmax_last_dim_backward_bf16_bytes(y, g, dx, oc, ld)
        })
    }

    pub fn softmax_last_dim_backward_f64(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        softmax_backward_typed("softmax_last_dim_backward_f64", inputs, outputs, layouts, params, |b, y, g, dx, oc, ld| {
            b.softmax_last_dim_backward_f64_bytes(y, g, dx, oc, ld)
        })
    }

    fn softmax_backward_typed<F>(
        label: &'static str,
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
        call: F,
    ) -> Result<()>
    where
        F: FnOnce(
            &fuel_vulkan_backend::VulkanBackend,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &mut fuel_vulkan_backend::VulkanStorageBytes,
            usize, usize,
        ) -> fuel_core_types::Result<()>,
    {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::softmax::{label}: expected 2 inputs (y, g) + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (outer_count, last_dim) = softmax_params(label, params)?;
        let y_guard = read_storage(&inputs[0])?;
        let g_guard = read_storage(&inputs[1])?;
        let mut dx_guard = write_storage(&outputs[0])?;
        let y = vulkan_input(&y_guard)?;
        let g = vulkan_input(&g_guard)?;
        let backend = y.backend().ok_or_else(|| {
            Error::Msg(format!(
                "vulkan_dispatch::softmax::{label}: y has no VulkanBackend handle.",
            )).bt()
        })?;
        let dx = vulkan_output(&mut dx_guard)?;
        call(backend, y, g, dx, outer_count, last_dim)
    }
}

// ===========================================================================
// Norm — RmsNorm last-dim f32 (V.2.C)
// ===========================================================================
//
// LayerNorm is V.3 (no Slang kernel yet). RmsNorm pulls
// `(outer_count, last_dim, eps)` from `OpParams::NormLastDim`.

pub mod norm {
    use super::*;

    // Internal helper: pull (outer_count, last_dim, eps) from the
    // OpParams::NormLastDim variant, returning a tagged error if the
    // executor passed the wrong params shape.
    fn norm_params(fn_name: &str, params: &OpParams) -> Result<(usize, usize, f64)> {
        match params {
            OpParams::NormLastDim { outer_count, last_dim, eps } => {
                Ok((*outer_count, *last_dim, *eps))
            }
            other => Err(Error::Msg(format!(
                "vulkan_dispatch::norm::{fn_name}: expected OpParams::NormLastDim, got {other:?}",
            )).bt()),
        }
    }

    pub fn rms_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::norm::rms_f32: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (outer_count, last_dim, eps) = norm_params("rms_f32", params)?;
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let a = vulkan_input(&in_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::norm::rms_f32: input has no VulkanBackend handle. \
                 Storages flowing through the pipelined-executor binding-table dispatch \
                 must come from alloc_bytes_handle / upload_bytes_handle."
                    .to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.rms_norm_last_dim_f32_bytes(a, out, outer_count, last_dim, eps)
    }

    pub fn rms_f16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::norm::rms_f16: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (outer_count, last_dim, eps) = norm_params("rms_f16", params)?;
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let a = vulkan_input(&in_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::norm::rms_f16: input has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.rms_norm_last_dim_f16_bytes(a, out, outer_count, last_dim, eps)
    }

    pub fn rms_bf16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::norm::rms_bf16: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (outer_count, last_dim, eps) = norm_params("rms_bf16", params)?;
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let a = vulkan_input(&in_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::norm::rms_bf16: input has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.rms_norm_last_dim_bf16_bytes(a, out, outer_count, last_dim, eps)
    }

    // ----- LayerNormLastDimBackward (V.3.G.layer_norm_bwd, 2026-05-30) -----
    // 2 inputs (x, g) → 1 output (dx). Reuses OpParams::NormLastDim.

    pub fn layer_norm_backward_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        layer_norm_bwd_typed("layer_norm_backward_f32", inputs, outputs, params, |b, x, g, dx, oc, ld, eps| {
            b.layer_norm_last_dim_backward_f32_bytes(x, g, dx, oc, ld, eps)
        })
    }
    pub fn layer_norm_backward_f16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        layer_norm_bwd_typed("layer_norm_backward_f16", inputs, outputs, params, |b, x, g, dx, oc, ld, eps| {
            b.layer_norm_last_dim_backward_f16_bytes(x, g, dx, oc, ld, eps)
        })
    }
    pub fn layer_norm_backward_bf16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        layer_norm_bwd_typed("layer_norm_backward_bf16", inputs, outputs, params, |b, x, g, dx, oc, ld, eps| {
            b.layer_norm_last_dim_backward_bf16_bytes(x, g, dx, oc, ld, eps)
        })
    }
    pub fn layer_norm_backward_f64(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        layer_norm_bwd_typed("layer_norm_backward_f64", inputs, outputs, params, |b, x, g, dx, oc, ld, eps| {
            b.layer_norm_last_dim_backward_f64_bytes(x, g, dx, oc, ld, eps)
        })
    }

    fn layer_norm_bwd_typed<F>(
        label: &'static str,
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        params: &OpParams,
        call: F,
    ) -> Result<()>
    where
        F: FnOnce(
            &fuel_vulkan_backend::VulkanBackend,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &mut fuel_vulkan_backend::VulkanStorageBytes,
            usize, usize, f64,
        ) -> fuel_core_types::Result<()>,
    {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::norm::{label}: expected 2 inputs (x, g) + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (outer_count, last_dim, eps) = norm_params(label, params)?;
        let x_guard = read_storage(&inputs[0])?;
        let g_guard = read_storage(&inputs[1])?;
        let mut dx_guard = write_storage(&outputs[0])?;
        let x = vulkan_input(&x_guard)?;
        let g = vulkan_input(&g_guard)?;
        let backend = x.backend().ok_or_else(|| {
            Error::Msg(format!(
                "vulkan_dispatch::norm::{label}: x has no VulkanBackend handle.",
            )).bt()
        })?;
        let dx = vulkan_output(&mut dx_guard)?;
        call(backend, x, g, dx, outer_count, last_dim, eps)
    }

    // ----- LayerNorm (V.3.G.layer_norm, 2026-05-30) -----

    pub fn layer_norm_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        layer_norm_typed("layer_norm_f32", inputs, outputs, params, |b, i, o, oc, ld, eps| {
            b.layer_norm_last_dim_f32_bytes(i, o, oc, ld, eps)
        })
    }
    pub fn layer_norm_f16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        layer_norm_typed("layer_norm_f16", inputs, outputs, params, |b, i, o, oc, ld, eps| {
            b.layer_norm_last_dim_f16_bytes(i, o, oc, ld, eps)
        })
    }
    pub fn layer_norm_bf16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        layer_norm_typed("layer_norm_bf16", inputs, outputs, params, |b, i, o, oc, ld, eps| {
            b.layer_norm_last_dim_bf16_bytes(i, o, oc, ld, eps)
        })
    }
    pub fn layer_norm_f64(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        layer_norm_typed("layer_norm_f64", inputs, outputs, params, |b, i, o, oc, ld, eps| {
            b.layer_norm_last_dim_f64_bytes(i, o, oc, ld, eps)
        })
    }

    fn layer_norm_typed<F>(
        label: &'static str,
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        params: &OpParams,
        call: F,
    ) -> Result<()>
    where
        F: FnOnce(
            &fuel_vulkan_backend::VulkanBackend,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &mut fuel_vulkan_backend::VulkanStorageBytes,
            usize, usize, f64,
        ) -> fuel_core_types::Result<()>,
    {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::norm::{label}: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (outer_count, last_dim, eps) = norm_params(label, params)?;
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let a = vulkan_input(&in_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(format!(
                "vulkan_dispatch::norm::{label}: input has no VulkanBackend handle.",
            )).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        call(backend, a, out, outer_count, last_dim, eps)
    }

    pub fn rms_f64(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::norm::rms_f64: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (outer_count, last_dim, eps) = norm_params("rms_f64", params)?;
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let a = vulkan_input(&in_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::norm::rms_f64: input has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.rms_norm_last_dim_f64_bytes(a, out, outer_count, last_dim, eps)
    }
}

// ===========================================================================
// Attention — RoPE f32 (V.2.C)
// ===========================================================================
//
// 3 storage inputs: x, cos, sin. The Vulkan kernel uses pre-computed
// cos/sin tables (unlike the CUDA baracuda path which rebuilds them
// from `seq + head_dim`). `OpParams::Rope` carries the geometry.

pub mod attention {
    use super::*;

    pub fn rope_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 3 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::attention::rope_f32: expected 3 inputs (x,cos,sin) + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        // Validate params shape (we don't actually need to read the
        // fields — the x-layout carries the geometry the kernel
        // wants — but mismatched params signal an executor bug).
        match params {
            OpParams::Rope { .. } => {}
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::attention::rope_f32: expected OpParams::Rope, got {:?}",
                    other,
                )).bt());
            }
        }
        if layouts.is_empty() {
            return Err(Error::Msg(
                "vulkan_dispatch::attention::rope_f32: layouts.len() = 0, need x-layout".into(),
            ).bt());
        }
        let x_guard = read_storage(&inputs[0])?;
        let cos_guard = read_storage(&inputs[1])?;
        let sin_guard = read_storage(&inputs[2])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let x = vulkan_input(&x_guard)?;
        let cos = vulkan_input(&cos_guard)?;
        let sin = vulkan_input(&sin_guard)?;
        let backend = x.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::attention::rope_f32: x has no VulkanBackend handle. \
                 Storages flowing through the pipelined-executor binding-table dispatch \
                 must come from alloc_bytes_handle / upload_bytes_handle."
                    .to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.rope_f32_bytes(x, cos, sin, out, &layouts[0])
    }

    pub fn rope_f16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        rope_typed("rope_f16", inputs, outputs, layouts, params, |b, x, c, s, o, l| {
            b.rope_f16_bytes(x, c, s, o, l)
        })
    }

    pub fn rope_bf16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        rope_typed("rope_bf16", inputs, outputs, layouts, params, |b, x, c, s, o, l| {
            b.rope_bf16_bytes(x, c, s, o, l)
        })
    }

    pub fn rope_f64(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        rope_typed("rope_f64", inputs, outputs, layouts, params, |b, x, c, s, o, l| {
            b.rope_f64_bytes(x, c, s, o, l)
        })
    }

    fn rope_typed<F>(
        label: &'static str,
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
        call: F,
    ) -> Result<()>
    where
        F: FnOnce(
            &fuel_vulkan_backend::VulkanBackend,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &mut fuel_vulkan_backend::VulkanStorageBytes,
            &Layout,
        ) -> fuel_core_types::Result<()>,
    {
        if inputs.len() != 3 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::attention::{label}: expected 3 inputs (x,cos,sin) + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        match params {
            OpParams::Rope { .. } => {}
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::attention::{label}: expected OpParams::Rope, got {other:?}",
                )).bt());
            }
        }
        if layouts.is_empty() {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::attention::{label}: layouts.len() = 0, need x-layout",
            )).bt());
        }
        let x_guard = read_storage(&inputs[0])?;
        let cos_guard = read_storage(&inputs[1])?;
        let sin_guard = read_storage(&inputs[2])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let x = vulkan_input(&x_guard)?;
        let cos = vulkan_input(&cos_guard)?;
        let sin = vulkan_input(&sin_guard)?;
        let backend = x.backend().ok_or_else(|| {
            Error::Msg(format!(
                "vulkan_dispatch::attention::{label}: x has no VulkanBackend handle.",
            )).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        call(backend, x, cos, sin, out, &layouts[0])
    }
}

// ===========================================================================
// PowI — y = x^exp, f32 (V.3.A.3)
// ===========================================================================

pub mod powi {
    use super::*;

    pub fn powi_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::powi::powi_f32: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        if layouts.is_empty() {
            return Err(Error::Msg(
                "vulkan_dispatch::powi::powi_f32: layouts.len() = 0, expected >= 1".into(),
            ).bt());
        }
        let exp = match params {
            OpParams::PowI { exp } => *exp,
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::powi::powi_f32: expected OpParams::PowI, got {:?}",
                    other,
                )).bt());
            }
        };
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let a = vulkan_input(&in_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::powi::powi_f32: input has no VulkanBackend handle. \
                 Storages flowing through the pipelined-executor binding-table dispatch \
                 must come from alloc_bytes_handle / upload_bytes_handle."
                    .to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.powi_f32_bytes(a, out, exp, &layouts[0])
    }
}

// ===========================================================================
// WriteSlice — in-place rectangular slab assign (V.3.J)
// ===========================================================================
//
// Backs `Op::WriteSlice` for persistent KV-cache writes (matches the
// CUDA pattern from baracuda alpha.29). The pipelined executor wires
// `inputs=[source]` + `outputs=[dest_arc]` where outputs[0]'s Arc IS
// the destination's Storage Arc (zero-copy in-place adoption). The
// wrapper holds a write lock on outputs[0] and a read lock on
// inputs[0]; the kernel mutates dest in place.
//
// 4-byte-keyed (f32/i32/u32 only in V.3.J — b2/b8 are follow-up).

pub mod write_slice {
    use super::*;

    macro_rules! vk_write_slice_wrapper {
        ($name:ident, $byte_width:expr $(,)?) => {
            pub fn $name(
                inputs: &[Arc<RwLock<Storage>>],
                outputs: &mut [Arc<RwLock<Storage>>],
                _layouts: &[Layout],
                params: &OpParams,
            ) -> Result<()> {
                if inputs.len() != 1 || outputs.len() != 1 {
                    return Err(Error::Msg(format!(
                        "vulkan_dispatch::write_slice::{}: expected 1 input + 1 output, got {} + {}",
                        stringify!($name), inputs.len(), outputs.len(),
                    )).bt());
                }
                let (dest_shape, ranges) = match params {
                    OpParams::WriteSlice { dest_shape, ranges } => (dest_shape, ranges),
                    other => {
                        return Err(Error::Msg(format!(
                            "vulkan_dispatch::write_slice::{}: expected OpParams::WriteSlice, got {:?}",
                            stringify!($name), other,
                        )).bt());
                    }
                };
                let rank = dest_shape.len();
                let mut source_shape = Vec::with_capacity(rank);
                let mut range_start = Vec::with_capacity(rank);
                for &(start, end) in ranges.iter() {
                    source_shape.push(end - start);
                    range_start.push(start);
                }

                let in_guard = read_storage(&inputs[0])?;
                let mut out_guard = write_storage(&outputs[0])?;
                let src_vk = vulkan_input(&in_guard)?;
                let backend = src_vk.backend().ok_or_else(|| {
                    Error::Msg(format!(
                        "vulkan_dispatch::write_slice::{}: src has no VulkanBackend handle. \
                         Storages flowing through the pipelined-executor binding-table dispatch \
                         must come from alloc_bytes_handle / upload_bytes_handle.",
                        stringify!($name),
                    )).bt()
                })?;
                let dst_vk = vulkan_output(&mut out_guard)?;
                backend.write_slice_bytes($byte_width, src_vk, dst_vk, dest_shape, &source_shape, &range_start)
            }
        };
    }

    vk_write_slice_wrapper!(write_slice_b1, 1);
    vk_write_slice_wrapper!(write_slice_b2, 2);
    vk_write_slice_wrapper!(write_slice_b4, 4);
    vk_write_slice_wrapper!(write_slice_b8, 8);
}

// ===========================================================================
// WriteSliceRotating — sliding-window KV cache writes (Phase C)
// ===========================================================================
//
// Mirrors the CUDA strategy in `baracuda_dispatch::write_slice_rotating`:
//   1. D2H the U32 position scalar (4 bytes) via
//      `VulkanBackend::download_bytes`.
//   2. Compute `wrapped_start = position % modulus`.
//   3. Split into up to two contiguous chunks across the ring
//      boundary.
//   4. Per chunk, extract the source prefix/suffix as a fresh
//      `VulkanStorageBytes` via `VulkanBackend::slot_copy_to_new_handle`
//      (vkCmdCopyBuffer-backed D2D), then reuse the existing
//      `VulkanBackend::write_slice_bytes` Slang kernels.
//
// v1 constraint: rotating axis must be axis 0 (matches Mistral /
// Phi-3 sliding-window K/V layout). Same v1 constraint as CPU + CUDA.

pub mod write_slice_rotating {
    use super::*;

    macro_rules! vk_write_slice_rotating_wrapper {
        ($name:ident, $byte_width:expr $(,)?) => {
            pub fn $name(
                inputs: &[Arc<RwLock<Storage>>],
                outputs: &mut [Arc<RwLock<Storage>>],
                _layouts: &[Layout],
                params: &OpParams,
            ) -> Result<()> {
                if inputs.len() != 2 || outputs.len() != 1 {
                    return Err(Error::Msg(format!(
                        "vulkan_dispatch::write_slice_rotating::{}: expected 2 inputs \
                         (source, position) + 1 output (dest), got {} + {}",
                        stringify!($name), inputs.len(), outputs.len(),
                    )).bt());
                }
                let (dest_shape, axis, modulus, ranges) = match params {
                    OpParams::WriteSliceRotating { dest_shape, axis, modulus, ranges } => {
                        (dest_shape, *axis, *modulus, ranges)
                    }
                    other => {
                        return Err(Error::Msg(format!(
                            "vulkan_dispatch::write_slice_rotating::{}: expected \
                             OpParams::WriteSliceRotating, got {:?}",
                            stringify!($name), other,
                        )).bt());
                    }
                };
                let rank = dest_shape.len();
                if ranges.len() != rank {
                    return Err(Error::Msg(format!(
                        "vulkan_dispatch::write_slice_rotating::{}: ranges.len() {} != dest rank {}",
                        stringify!($name), ranges.len(), rank,
                    )).bt());
                }

                // Position is rank-0 U32, stored on the same device as
                // the source. Grab a backend handle from the position
                // storage so we can do the D2H + per-chunk D2D.
                let pos_guard = read_storage(&inputs[1])?;
                let pos_vk = vulkan_input(&pos_guard)?;
                let backend = pos_vk.backend().ok_or_else(|| {
                    Error::Msg(format!(
                        "vulkan_dispatch::write_slice_rotating::{}: position storage has no \
                         VulkanBackend handle",
                        stringify!($name),
                    )).bt()
                })?;
                let pos_bytes = backend.download_bytes(pos_vk)?;
                if pos_bytes.len() < 4 {
                    return Err(Error::Msg(format!(
                        "vulkan_dispatch::write_slice_rotating::{}: position storage has {} bytes, need >= 4",
                        stringify!($name), pos_bytes.len(),
                    )).bt());
                }
                let position = u32::from_ne_bytes([
                    pos_bytes[0], pos_bytes[1], pos_bytes[2], pos_bytes[3],
                ]) as usize;
                let wrapped_start = position % modulus;

                // Derive slab_shape from ranges.
                let mut slab_shape: Vec<usize> = Vec::with_capacity(rank);
                for (i, &(start, end)) in ranges.iter().enumerate() {
                    if end < start {
                        return Err(Error::Msg(format!(
                            "vulkan_dispatch::write_slice_rotating::{}: ranges[{}] = ({}, {}) has end < start",
                            stringify!($name), i, start, end,
                        )).bt());
                    }
                    let slab = end - start;
                    if i == axis && slab > modulus {
                        return Err(Error::Msg(format!(
                            "vulkan_dispatch::write_slice_rotating::{}: rotating-axis slab {} > modulus {}",
                            stringify!($name), slab, modulus,
                        )).bt());
                    }
                    slab_shape.push(slab);
                }
                let slab_axis_len = slab_shape[axis];
                let slab_elems: usize = slab_shape.iter().copied().product();
                if slab_elems == 0 {
                    return Ok(());
                }

                let first_len = slab_axis_len.min(modulus - wrapped_start);
                let second_len = slab_axis_len - first_len;

                // Outer/axis/inner decomposition for strided source extract.
                // For axis 0, outer_count is 1 and the strided extract
                // reduces to a single contiguous BufferCopy.
                let outer_count: usize = slab_shape[..axis].iter().copied().product();
                let inner_per_row: usize = slab_shape[axis + 1..].iter().copied().product();
                let row_bytes = inner_per_row * $byte_width;
                let stride_bytes = slab_axis_len * row_bytes;

                let src_guard = read_storage(&inputs[0])?;
                let src_vk = vulkan_input(&src_guard)?;

                if first_len > 0 {
                    let chunk_row_bytes = first_len * row_bytes;
                    let src_first = backend.extract_strided_to_new_handle(
                        src_vk, outer_count, stride_bytes, /* offset_in_outer */ 0, chunk_row_bytes,
                    )?;
                    let mut sub_source_shape = slab_shape.clone();
                    sub_source_shape[axis] = first_len;
                    let mut sub_range_start: Vec<usize> = ranges.iter().map(|r| r.0).collect();
                    sub_range_start[axis] = wrapped_start;

                    let mut out_guard = write_storage(&outputs[0])?;
                    let dst_vk = vulkan_output(&mut out_guard)?;
                    backend.write_slice_bytes(
                        $byte_width, &src_first, dst_vk,
                        dest_shape, &sub_source_shape, &sub_range_start,
                    )?;
                }
                if second_len > 0 {
                    let chunk_row_bytes = second_len * row_bytes;
                    let offset_in_outer = first_len * row_bytes;
                    let src_second = backend.extract_strided_to_new_handle(
                        src_vk, outer_count, stride_bytes, offset_in_outer, chunk_row_bytes,
                    )?;
                    let mut sub_source_shape = slab_shape.clone();
                    sub_source_shape[axis] = second_len;
                    let mut sub_range_start: Vec<usize> = ranges.iter().map(|r| r.0).collect();
                    sub_range_start[axis] = 0;

                    let mut out_guard = write_storage(&outputs[0])?;
                    let dst_vk = vulkan_output(&mut out_guard)?;
                    backend.write_slice_bytes(
                        $byte_width, &src_second, dst_vk,
                        dest_shape, &sub_source_shape, &sub_range_start,
                    )?;
                }
                Ok(())
            }
        };
    }

    vk_write_slice_rotating_wrapper!(write_slice_rotating_b1, 1);
    vk_write_slice_rotating_wrapper!(write_slice_rotating_b2, 2);
    vk_write_slice_rotating_wrapper!(write_slice_rotating_b4, 4);
    vk_write_slice_rotating_wrapper!(write_slice_rotating_b8, 8);
}

// ===========================================================================
// Cast — f32↔f16, f32↔bf16 (V.3.B)
// ===========================================================================
//
// One wrapper handles all 4 supported (src, dst) pairs by inspecting
// Storage.dtype. Element count derived from src bytes / src elem-size.
// Requires `n` even (half-precision dtypes are u32-packed 2-per-word);
// odd-count tensors return Err and the route picker falls back to CPU.

pub mod cast {
    use super::*;

    pub fn cast_f32_f64(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        _params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::cast::cast_f32_f64: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let src_dtype = in_guard.dtype;
        let dst_dtype = out_guard.dtype;
        let a = vulkan_input(&in_guard)?;
        let n_bytes = a.len_bytes();
        let src_elem = match src_dtype {
            DType::F32 => 4,
            DType::F64 => 8,
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::cast::cast_f32_f64: unsupported src dtype {other:?}",
                )).bt());
            }
        };
        if n_bytes % src_elem != 0 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::cast::cast_f32_f64: input bytes {n_bytes} not a multiple of \
                 src elem size {src_elem} ({src_dtype:?})",
            )).bt());
        }
        let n = n_bytes / src_elem;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::cast::cast_f32_f64: input has no VulkanBackend handle."
                    .to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.cast_f32_f64_bytes(a, out, n, src_dtype, dst_dtype)
    }

    pub fn cast_f32_half(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        _params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::cast::cast_f32_half: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let src_dtype = in_guard.dtype;
        let dst_dtype = out_guard.dtype;
        let a = vulkan_input(&in_guard)?;
        let n_bytes = a.len_bytes();
        let src_elem = match src_dtype {
            DType::F32  => 4,
            DType::F16  => 2,
            DType::BF16 => 2,
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::cast::cast_f32_half: unsupported src dtype {other:?}",
                )).bt());
            }
        };
        if n_bytes % src_elem != 0 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::cast::cast_f32_half: input bytes {n_bytes} not a multiple of \
                 src elem size {src_elem} ({src_dtype:?})",
            )).bt());
        }
        let n = n_bytes / src_elem;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::cast::cast_f32_half: input has no VulkanBackend handle. \
                 Storages flowing through the pipelined-executor binding-table dispatch \
                 must come from alloc_bytes_handle / upload_bytes_handle."
                    .to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.cast_f32_bytes(a, out, n, src_dtype, dst_dtype)
    }
}

// ===========================================================================
// Clamp — y = clamp(x, lo, hi), f32 (V.3.A.1)
// ===========================================================================

pub mod clamp {
    use super::*;

    pub fn clamp_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::clamp::clamp_f32: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        if layouts.is_empty() {
            return Err(Error::Msg(
                "vulkan_dispatch::clamp::clamp_f32: layouts.len() = 0, expected >= 1".into(),
            ).bt());
        }
        let (lo, hi) = match params {
            OpParams::Clamp { min, max } => (*min, *max),
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::clamp::clamp_f32: expected OpParams::Clamp, got {:?}",
                    other,
                )).bt());
            }
        };
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let a = vulkan_input(&in_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::clamp::clamp_f32: input has no VulkanBackend handle. \
                 Storages flowing through the pipelined-executor binding-table dispatch \
                 must come from alloc_bytes_handle / upload_bytes_handle."
                    .to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.clamp_f32_bytes(a, out, lo, hi, &layouts[0])
    }
}

// ===========================================================================
// Affine — y = mul*x + add, f32 (V.2.E)
// ===========================================================================

pub mod affine {
    use super::*;

    pub fn affine_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::affine::affine_f32: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        if layouts.is_empty() {
            return Err(Error::Msg(
                "vulkan_dispatch::affine::affine_f32: layouts.len() = 0, expected >= 1".into(),
            ).bt());
        }
        let (mul, add) = match params {
            OpParams::Affine { mul, add } => (*mul, *add),
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::affine::affine_f32: expected OpParams::Affine, got {:?}",
                    other,
                )).bt());
            }
        };
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let a = vulkan_input(&in_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::affine::affine_f32: input has no VulkanBackend handle. \
                 Storages flowing through the pipelined-executor binding-table dispatch \
                 must come from alloc_bytes_handle / upload_bytes_handle."
                    .to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.affine_f32_bytes(a, out, mul, add, &layouts[0])
    }

    pub fn affine_f64(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        affine_typed("affine_f64", inputs, outputs, layouts, params,
            |b, a, out, mul, add, layout| b.affine_f64_bytes(a, out, mul, add, layout))
    }

    pub fn affine_f16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        affine_typed("affine_f16", inputs, outputs, layouts, params,
            |b, a, out, mul, add, layout| b.affine_f16_bytes(a, out, mul, add, layout))
    }

    pub fn affine_bf16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        affine_typed("affine_bf16", inputs, outputs, layouts, params,
            |b, a, out, mul, add, layout| b.affine_bf16_bytes(a, out, mul, add, layout))
    }

    fn affine_typed<F>(
        debug_name: &'static str,
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
        f: F,
    ) -> Result<()>
    where
        F: FnOnce(
            &fuel_vulkan_backend::VulkanBackend,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &mut fuel_vulkan_backend::VulkanStorageBytes,
            f64, f64, &Layout,
        ) -> Result<()>,
    {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::affine::{debug_name}: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        if layouts.is_empty() {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::affine::{debug_name}: layouts.len() = 0",
            )).bt());
        }
        let (mul, add) = match params {
            OpParams::Affine { mul, add } => (*mul, *add),
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::affine::{debug_name}: expected OpParams::Affine, got {other:?}",
                )).bt());
            }
        };
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let a = vulkan_input(&in_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(format!(
                "vulkan_dispatch::affine::{debug_name}: input has no VulkanBackend handle.",
            )).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        f(backend, a, out, mul, add, &layouts[0])
    }
}

// ===========================================================================
// MatMul — f32 (V.2.D)
// ===========================================================================
//
// Selects among matvec (m==1) / reg-tile (m<32) / tiled (m>=32)
// kernels via `VulkanBackend::matmul_f32_bytes`. Mixed-bf16 and
// cooperative-matrix paths are deferred to V.3. GQA broadcast is
// honored (lhs_batch > rhs_batch with even divisibility).

pub mod matmul {
    use super::*;

    pub fn matmul_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::matmul::matmul_f32: expected 2 inputs + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (lhs_batch_dims, rhs_batch_dims, m, n, k) = match params {
            OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k } => {
                (lhs_batch_dims.clone(), rhs_batch_dims.clone(), *m, *n, *k)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::matmul::matmul_f32: expected OpParams::Matmul, got {:?}",
                    other,
                )).bt());
            }
        };
        let lhs_guard = read_storage(&inputs[0])?;
        let rhs_guard = read_storage(&inputs[1])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let lhs = vulkan_input(&lhs_guard)?;
        let rhs = vulkan_input(&rhs_guard)?;
        let backend = lhs.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::matmul::matmul_f32: lhs has no VulkanBackend handle. \
                 Storages flowing through the pipelined-executor binding-table dispatch \
                 must come from alloc_bytes_handle / upload_bytes_handle."
                    .to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.matmul_f32_bytes(lhs, rhs, out, &lhs_batch_dims, &rhs_batch_dims, m, n, k)
    }

    /// Mixed-precision matmul: f32 LHS × bf16 RHS → f32 output.
    /// Vulkan-specific decision-point (CUDA registers full-bf16
    /// `[bf16, bf16, bf16]` instead). The route picker prefers this
    /// when the input dtypes match exactly; otherwise falls back to
    /// f32 matmul (after a Cast).
    pub fn matmul_f32_bf16_b(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::matmul::matmul_f32_bf16_b: expected 2 inputs + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (lhs_batch_dims, rhs_batch_dims, m, n, k) = match params {
            OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k } => {
                (lhs_batch_dims.clone(), rhs_batch_dims.clone(), *m, *n, *k)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::matmul::matmul_f32_bf16_b: expected OpParams::Matmul, got {:?}",
                    other,
                )).bt());
            }
        };
        let lhs_guard = read_storage(&inputs[0])?;
        let rhs_guard = read_storage(&inputs[1])?;
        let mut out_guard = write_storage(&outputs[0])?;
        if lhs_guard.dtype != DType::F32 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::matmul::matmul_f32_bf16_b: lhs must be F32, got {:?}",
                lhs_guard.dtype,
            )).bt());
        }
        if rhs_guard.dtype != DType::BF16 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::matmul::matmul_f32_bf16_b: rhs must be BF16, got {:?}",
                rhs_guard.dtype,
            )).bt());
        }
        let lhs = vulkan_input(&lhs_guard)?;
        let rhs = vulkan_input(&rhs_guard)?;
        let backend = lhs.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::matmul::matmul_f32_bf16_b: lhs has no VulkanBackend handle. \
                 Storages flowing through the pipelined-executor binding-table dispatch \
                 must come from alloc_bytes_handle / upload_bytes_handle."
                    .to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.matmul_f32_bf16_b_bytes(
            lhs, rhs, out, &lhs_batch_dims, &rhs_batch_dims, m, n, k,
        )
    }

    /// MatMul bf16 × bf16 → f32 via cooperative-matrix tensor cores
    /// (coop[3] tile: A=f16, B=f16, C=f32, R=f32; both inputs are
    /// stored as bf16 and downcast bf16→f16 on shared-mem load).
    /// COOP-ONLY — bails on small shapes; route picker should fall
    /// through to a cast-and-f32-matmul alternative.
    pub fn matmul_bf16_bf16_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::matmul::matmul_bf16_bf16_f32: expected 2 inputs + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (lhs_batch_dims, rhs_batch_dims, m, n, k) = match params {
            OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k } => {
                (lhs_batch_dims.clone(), rhs_batch_dims.clone(), *m, *n, *k)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::matmul::matmul_bf16_bf16_f32: expected OpParams::Matmul, got {other:?}",
                )).bt());
            }
        };
        let lhs_guard = read_storage(&inputs[0])?;
        let rhs_guard = read_storage(&inputs[1])?;
        let mut out_guard = write_storage(&outputs[0])?;
        if lhs_guard.dtype != DType::BF16 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::matmul::matmul_bf16_bf16_f32: lhs must be BF16, got {:?}",
                lhs_guard.dtype,
            )).bt());
        }
        if rhs_guard.dtype != DType::BF16 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::matmul::matmul_bf16_bf16_f32: rhs must be BF16, got {:?}",
                rhs_guard.dtype,
            )).bt());
        }
        let lhs = vulkan_input(&lhs_guard)?;
        let rhs = vulkan_input(&rhs_guard)?;
        let backend = lhs.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::matmul::matmul_bf16_bf16_f32: lhs has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.matmul_bf16_bf16_f32_bytes(
            lhs, rhs, out, &lhs_batch_dims, &rhs_batch_dims, m, n, k,
        )
    }

    /// MatMul f16 × f16 → f32 via cooperative-matrix tensor cores.
    /// Same constraints as the bf16 sibling.
    pub fn matmul_f16_f16_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::matmul::matmul_f16_f16_f32: expected 2 inputs + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (lhs_batch_dims, rhs_batch_dims, m, n, k) = match params {
            OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k } => {
                (lhs_batch_dims.clone(), rhs_batch_dims.clone(), *m, *n, *k)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::matmul::matmul_f16_f16_f32: expected OpParams::Matmul, got {other:?}",
                )).bt());
            }
        };
        let lhs_guard = read_storage(&inputs[0])?;
        let rhs_guard = read_storage(&inputs[1])?;
        let mut out_guard = write_storage(&outputs[0])?;
        if lhs_guard.dtype != DType::F16 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::matmul::matmul_f16_f16_f32: lhs must be F16, got {:?}",
                lhs_guard.dtype,
            )).bt());
        }
        if rhs_guard.dtype != DType::F16 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::matmul::matmul_f16_f16_f32: rhs must be F16, got {:?}",
                rhs_guard.dtype,
            )).bt());
        }
        let lhs = vulkan_input(&lhs_guard)?;
        let rhs = vulkan_input(&rhs_guard)?;
        let backend = lhs.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::matmul::matmul_f16_f16_f32: lhs has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.matmul_f16_f16_f32_bytes(
            lhs, rhs, out, &lhs_batch_dims, &rhs_batch_dims, m, n, k,
        )
    }

    /// MatMul bf16 × bf16 → bf16 (downcast store). Coop kernel with
    /// f32 accumulator + shared-memory staging + per-lane f32→bf16
    /// conversion. Closes the bf16 inference chain.
    pub fn matmul_bf16_bf16_bf16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::matmul::matmul_bf16_bf16_bf16: expected 2 inputs + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (lhs_batch_dims, rhs_batch_dims, m, n, k) = match params {
            OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k } => {
                (lhs_batch_dims.clone(), rhs_batch_dims.clone(), *m, *n, *k)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::matmul::matmul_bf16_bf16_bf16: expected OpParams::Matmul, got {other:?}",
                )).bt());
            }
        };
        let lhs_guard = read_storage(&inputs[0])?;
        let rhs_guard = read_storage(&inputs[1])?;
        let mut out_guard = write_storage(&outputs[0])?;
        if lhs_guard.dtype != DType::BF16 || rhs_guard.dtype != DType::BF16 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::matmul::matmul_bf16_bf16_bf16: lhs/rhs must be BF16, got ({:?}, {:?})",
                lhs_guard.dtype, rhs_guard.dtype,
            )).bt());
        }
        let lhs = vulkan_input(&lhs_guard)?;
        let rhs = vulkan_input(&rhs_guard)?;
        let backend = lhs.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::matmul::matmul_bf16_bf16_bf16: lhs has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.matmul_bf16_bf16_bf16_bytes(
            lhs, rhs, out, &lhs_batch_dims, &rhs_batch_dims, m, n, k,
        )
    }

    /// MatMul f16 × f16 → f16 (downcast store). Native float16_t
    /// inputs; f32 accumulator + shared-mem staging + per-lane
    /// `float16BitsToUint16` pack.
    pub fn matmul_f16_f16_f16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::matmul::matmul_f16_f16_f16: expected 2 inputs + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (lhs_batch_dims, rhs_batch_dims, m, n, k) = match params {
            OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k } => {
                (lhs_batch_dims.clone(), rhs_batch_dims.clone(), *m, *n, *k)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::matmul::matmul_f16_f16_f16: expected OpParams::Matmul, got {other:?}",
                )).bt());
            }
        };
        let lhs_guard = read_storage(&inputs[0])?;
        let rhs_guard = read_storage(&inputs[1])?;
        let mut out_guard = write_storage(&outputs[0])?;
        if lhs_guard.dtype != DType::F16 || rhs_guard.dtype != DType::F16 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::matmul::matmul_f16_f16_f16: lhs/rhs must be F16, got ({:?}, {:?})",
                lhs_guard.dtype, rhs_guard.dtype,
            )).bt());
        }
        let lhs = vulkan_input(&lhs_guard)?;
        let rhs = vulkan_input(&rhs_guard)?;
        let backend = lhs.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::matmul::matmul_f16_f16_f16: lhs has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.matmul_f16_f16_f16_bytes(
            lhs, rhs, out, &lhs_batch_dims, &rhs_batch_dims, m, n, k,
        )
    }
}

// ===========================================================================
// Concat — f32, binary (V.2.D)
// ===========================================================================
//
// Vulkan's legacy `concat_along_dim` kernel is 2-input. Fuel's
// `OpKind::Concat` is N-input. This wrapper handles N == 2 only —
// for N > 2 it errors and the route picker falls back to a CPU /
// CUDA alternative. Chain-for-N>2 is V.3 work.
//
// The concat `dim` is recovered from `OpParams::Concat.outer_count`
// + the first input's layout (outer_count = product of dims before
// the concat axis).

pub mod gather {
    use super::*;

    /// Gather along `dim`. 2 inputs (src, U32 indices) + 1 output.
    /// Output dtype determines byte width.
    pub fn gather(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::gather::gather: expected 2 inputs (src, indices) + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (source_shape, output_shape, dim) = match params {
            OpParams::Gather { source_shape, output_shape, dim } => {
                (source_shape.clone(), output_shape.clone(), *dim)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::gather::gather: expected OpParams::Gather, got {other:?}",
                )).bt());
            }
        };
        let src_guard = read_storage(&inputs[0])?;
        let idx_guard = read_storage(&inputs[1])?;
        if idx_guard.dtype != DType::U32 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::gather::gather: indices must be U32, got {:?}",
                idx_guard.dtype,
            )).bt());
        }
        let mut out_guard = write_storage(&outputs[0])?;
        let elem_bytes = out_guard.dtype.size_in_bytes();
        let src = vulkan_input(&src_guard)?;
        let indices = vulkan_input(&idx_guard)?;
        let backend = src.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::gather::gather: src has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.gather_bytes(src, indices, out, &source_shape, &output_shape, dim, elem_bytes)
    }
}

pub mod masked_fill {
    use super::*;

    /// MaskedFill — 2 inputs (input, mask) + 1 output. mask is U8.
    /// Byte width is taken from the OUTPUT dtype size; same dispatch
    /// shim handles all dtypes.
    pub fn masked_fill(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::masked_fill::masked_fill: expected 2 inputs (input, mask) + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let fill_bytes = match params {
            OpParams::MaskedFill { fill_bytes } => fill_bytes.clone(),
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::masked_fill::masked_fill: expected OpParams::MaskedFill, got {other:?}",
                )).bt());
            }
        };
        let in_guard = read_storage(&inputs[0])?;
        let mask_guard = read_storage(&inputs[1])?;
        if mask_guard.dtype != DType::U8 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::masked_fill::masked_fill: mask must be U8, got {:?}",
                mask_guard.dtype,
            )).bt());
        }
        let mut out_guard = write_storage(&outputs[0])?;
        let elem_bytes = out_guard.dtype.size_in_bytes();
        let a = vulkan_input(&in_guard)?;
        let mask = vulkan_input(&mask_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::masked_fill::masked_fill: input has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        let n_elem = a.len_bytes() / elem_bytes;
        backend.masked_fill_bytes(a, mask, out, n_elem, elem_bytes, &fill_bytes)
    }
}

pub mod pad {
    use super::*;

    /// PadBackward — 1 input (grad_out) + 1 output (grad_in). Reads
    /// geometry from OpParams::PadBackward. Currently handles
    /// mode_tag == 0 (constant) only; reflect/replicate need atomic
    /// float add (not yet wired) and fall through to CPU.
    pub fn pad_backward(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::pad::pad_backward: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (in_shape, out_shape, padding, mode_tag) = match params {
            OpParams::PadBackward { in_shape, out_shape, padding, mode_tag } => {
                (in_shape, out_shape, padding, *mode_tag)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::pad::pad_backward: expected OpParams::PadBackward, got {other:?}",
                )).bt());
            }
        };
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let dtype = out_guard.dtype;
        let elem_bytes = dtype.size_in_bytes();
        let go = vulkan_input(&in_guard)?;
        let backend = go.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::pad::pad_backward: grad_out has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let gi = vulkan_output(&mut out_guard)?;
        let left_pad: Vec<usize> = padding.iter().map(|&(b, _)| b).collect();
        match mode_tag {
            0 => backend.pad_backward_const_bytes(go, gi, in_shape, out_shape, &left_pad, elem_bytes),
            1 | 2 => match dtype {
                DType::F32 | DType::F64 | DType::BF16 | DType::F16 => {
                    backend.pad_backward_atomic_bytes(dtype, go, gi, in_shape, out_shape, &left_pad, mode_tag)
                }
                _ => Err(Error::Msg(format!(
                    "vulkan_dispatch::pad::pad_backward: mode_tag {mode_tag} only \
                     supports F32/F64/BF16/F16 on Vulkan (atomic CAS path); got {dtype:?}",
                )).bt()),
            },
            other => Err(Error::Msg(format!(
                "vulkan_dispatch::pad::pad_backward: unknown mode_tag {other}",
            )).bt()),
        }
    }

    /// Pad — 1 input + 1 output. Reads geometry from OpParams::Pad.
    /// `mode_tag == 0` (constant) is the only mode this dispatch
    /// shim handles; other modes fall through to CPU.
    /// Byte width is taken from the OUTPUT dtype's size.
    pub fn pad_const(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::pad::pad_const: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (in_shape, out_shape, padding, mode_tag, fill_bytes) = match params {
            OpParams::Pad { in_shape, out_shape, padding, mode_tag, fill_bytes } => {
                (in_shape, out_shape, padding, *mode_tag, fill_bytes)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::pad::pad_const: expected OpParams::Pad, got {other:?}",
                )).bt());
            }
        };
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let elem_bytes = out_guard.dtype.size_in_bytes();
        let a = vulkan_input(&in_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::pad::pad_const: input has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        let left_pad: Vec<usize> = padding.iter().map(|&(b, _)| b).collect();
        match mode_tag {
            0 => backend.pad_const_bytes(a, out, in_shape, out_shape, &left_pad, elem_bytes, fill_bytes),
            1 => backend.pad_reflect_bytes(a, out, in_shape, out_shape, &left_pad, elem_bytes),
            2 => backend.pad_replicate_bytes(a, out, in_shape, out_shape, &left_pad, elem_bytes),
            other => Err(Error::Msg(format!(
                "vulkan_dispatch::pad::pad_const: unknown mode_tag {other}",
            )).bt()),
        }
    }
}

pub mod concat {
    use super::*;
    use fuel_memory::BackendStorage;
    use fuel_core_types::Shape;

    /// Generic concat dispatcher. Takes the per-pair concat call
    /// (which knows the source/destination dtype) plus the per-element
    /// byte size for sizing intermediates in the N>2 chain. The same
    /// N=1/N=2/N>2 shape works across all float dtypes.
    fn concat_typed<F>(
        label: &'static str,
        elem_bytes: usize,
        dtype: DType,
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
        pair_call: F,
    ) -> Result<()>
    where
        F: Fn(
            &fuel_vulkan_backend::VulkanBackend,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &mut fuel_vulkan_backend::VulkanStorageBytes,
            usize,
            &Layout,
            &Layout,
        ) -> fuel_core_types::Result<()>,
    {
        if outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::concat::{label}: expected 1 output, got {}",
                outputs.len(),
            )).bt());
        }
        if inputs.is_empty() {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::concat::{label}: 0 inputs",
            )).bt());
        }
        let (outer_count, input_dim_sizes, inner_count, dim) = match params {
            OpParams::Concat { outer_count, input_dim_sizes, inner_count, axis } => {
                (*outer_count, input_dim_sizes.clone(), *inner_count, *axis)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::concat::{label}: expected OpParams::Concat, got {other:?}",
                )).bt());
            }
        };
        if input_dim_sizes.len() != inputs.len() {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::concat::{label}: OpParams declares {} inputs but work item carries {}",
                input_dim_sizes.len(), inputs.len(),
            )).bt());
        }
        if layouts.len() < inputs.len() {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::concat::{label}: layouts.len() = {} < inputs.len() = {}",
                layouts.len(), inputs.len(),
            )).bt());
        }

        let n_inputs = inputs.len();

        if n_inputs == 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::concat::{label}: N=1 concat unsupported \
                 (should be elided at graph level)",
            )).bt());
        }

        if n_inputs == 2 {
            let a_guard = read_storage(&inputs[0])?;
            let b_guard = read_storage(&inputs[1])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let a = vulkan_input(&a_guard)?;
            let b = vulkan_input(&b_guard)?;
            let backend = a.backend().ok_or_else(|| {
                Error::Msg(format!(
                    "vulkan_dispatch::concat::{label}: a has no VulkanBackend handle.",
                )).bt()
            })?;
            let out = vulkan_output(&mut out_guard)?;
            return pair_call(backend, a, b, out, dim, &layouts[0], &layouts[1]);
        }

        // N > 2: chain pair-wise.
        let in_guards: Vec<_> = inputs.iter().map(read_storage).collect::<Result<Vec<_>>>()?;
        let mut in_vk: Vec<&fuel_vulkan_backend::VulkanStorageBytes> = Vec::with_capacity(n_inputs);
        for g in &in_guards {
            in_vk.push(vulkan_input(g)?);
        }
        let backend = in_vk[0].backend().ok_or_else(|| {
            Error::Msg(format!(
                "vulkan_dispatch::concat::{label}: inputs[0] has no VulkanBackend handle.",
            )).bt()
        })?.clone();

        let template_dims: Vec<usize> = layouts[0].shape().dims().to_vec();
        let make_layout = |cum_dim: usize| -> Layout {
            let mut dims = template_dims.clone();
            dims[dim] = cum_dim;
            Layout::contiguous(Shape::from_dims(&dims))
        };

        let mut acc_dim = input_dim_sizes[0] + input_dim_sizes[1];
        let acc_elems = outer_count * acc_dim * inner_count;
        let acc_bytes = backend.alloc_bytes_handle(acc_elems * elem_bytes).map_err(|e| {
            Error::Msg(format!(
                "vulkan_dispatch::concat::{label}: alloc intermediate failed: {e}",
            )).bt()
        })?;
        let mut acc_storage = Storage::new(BackendStorage::Vulkan(acc_bytes), dtype);
        {
            let acc_vk = match &mut acc_storage.inner {
                BackendStorage::Vulkan(v) => v,
                _ => unreachable!("just allocated as Vulkan"),
            };
            pair_call(&backend, in_vk[0], in_vk[1], acc_vk, dim, &layouts[0], &layouts[1])?;
        }

        for i in 2..(n_inputs - 1) {
            let new_acc_dim = acc_dim + input_dim_sizes[i];
            let new_elems = outer_count * new_acc_dim * inner_count;
            let new_bytes = backend.alloc_bytes_handle(new_elems * elem_bytes).map_err(|e| {
                Error::Msg(format!(
                    "vulkan_dispatch::concat::{label}: alloc intermediate {i} failed: {e}",
                )).bt()
            })?;
            let mut new_storage = Storage::new(BackendStorage::Vulkan(new_bytes), dtype);

            let acc_layout = make_layout(acc_dim);
            let acc_vk_ref = match &acc_storage.inner {
                BackendStorage::Vulkan(v) => v,
                _ => unreachable!(),
            };
            let new_vk = match &mut new_storage.inner {
                BackendStorage::Vulkan(v) => v,
                _ => unreachable!(),
            };
            pair_call(&backend, acc_vk_ref, in_vk[i], new_vk, dim, &acc_layout, &layouts[i])?;

            acc_storage = new_storage;
            acc_dim = new_acc_dim;
        }

        let acc_layout = make_layout(acc_dim);
        let acc_vk_ref = match &acc_storage.inner {
            BackendStorage::Vulkan(v) => v,
            _ => unreachable!(),
        };
        let mut out_guard = write_storage(&outputs[0])?;
        let out_vk = vulkan_output(&mut out_guard)?;
        pair_call(&backend, acc_vk_ref, in_vk[n_inputs - 1], out_vk, dim,
            &acc_layout, &layouts[n_inputs - 1])
    }

    pub fn concat_f16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        concat_typed(
            "concat_f16", 2, DType::F16,
            inputs, outputs, layouts, params,
            |b, a, bb, o, dim, la, lb| b.concat_along_dim_f16_bytes(a, bb, o, dim, la, lb),
        )
    }

    pub fn concat_bf16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        concat_typed(
            "concat_bf16", 2, DType::BF16,
            inputs, outputs, layouts, params,
            |b, a, bb, o, dim, la, lb| b.concat_along_dim_bf16_bytes(a, bb, o, dim, la, lb),
        )
    }

    pub fn concat_f64(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        concat_typed(
            "concat_f64", 8, DType::F64,
            inputs, outputs, layouts, params,
            |b, a, bb, o, dim, la, lb| b.concat_along_dim_f64_bytes(a, bb, o, dim, la, lb),
        )
    }

    pub fn concat_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::concat::concat_f32: expected 1 output, got {}",
                outputs.len(),
            )).bt());
        }
        if inputs.is_empty() {
            return Err(Error::Msg(
                "vulkan_dispatch::concat::concat_f32: 0 inputs".to_string(),
            ).bt());
        }
        let (outer_count, input_dim_sizes, inner_count, dim) = match params {
            OpParams::Concat { outer_count, input_dim_sizes, inner_count, axis } => {
                (*outer_count, input_dim_sizes.clone(), *inner_count, *axis)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::concat::concat_f32: expected OpParams::Concat, got {:?}",
                    other,
                )).bt());
            }
        };
        if input_dim_sizes.len() != inputs.len() {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::concat::concat_f32: OpParams declares {} inputs but work item carries {}",
                input_dim_sizes.len(), inputs.len(),
            )).bt());
        }
        if layouts.len() < inputs.len() {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::concat::concat_f32: layouts.len() = {} < inputs.len() = {}",
                layouts.len(), inputs.len(),
            )).bt());
        }

        let n_inputs = inputs.len();

        // N == 1 degenerate: copy input → out.
        if n_inputs == 1 {
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let a = vulkan_input(&in_guard)?;
            let _backend = a.backend().ok_or_else(|| {
                Error::Msg(
                    "vulkan_dispatch::concat::concat_f32: input has no VulkanBackend handle."
                        .to_string(),
                ).bt()
            })?;
            let out = vulkan_output(&mut out_guard)?;
            let a_bytes = a.buffer_opt().ok_or_else(|| Error::Msg(
                "concat_f32 N=1: a is host-evicted".into()).bt())?;
            let out_buf = out.buffer_opt().ok_or_else(|| Error::Msg(
                "concat_f32 N=1: out is host-evicted".into()).bt())?;
            // Equal sizes — direct memcpy via the legacy queue.one_shot.
            // For now, error: N=1 should be optimized at the graph level
            // (skip the concat entirely), not via dispatch.
            let _ = (a_bytes, out_buf);
            return Err(Error::Msg(
                "vulkan_dispatch::concat::concat_f32: N=1 concat unsupported \
                 (should be elided at graph level)".into(),
            ).bt());
        }

        // Fast path: N == 2. Direct single-call.
        if n_inputs == 2 {
            let a_guard = read_storage(&inputs[0])?;
            let b_guard = read_storage(&inputs[1])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let a = vulkan_input(&a_guard)?;
            let b = vulkan_input(&b_guard)?;
            let backend = a.backend().ok_or_else(|| {
                Error::Msg(
                    "vulkan_dispatch::concat::concat_f32: a has no VulkanBackend handle."
                        .to_string(),
                ).bt()
            })?;
            let out = vulkan_output(&mut out_guard)?;
            return backend.concat_along_dim_f32_bytes(a, b, out, dim, &layouts[0], &layouts[1]);
        }

        // N > 2: chain pair-wise via intermediate byte-storages. The
        // accumulator grows by `input_dim_sizes[i]` at each step.
        // We acquire all input guards up front to satisfy the borrow
        // checker (concat reads inputs sequentially).
        let in_guards: Vec<_> = inputs.iter().map(read_storage).collect::<Result<Vec<_>>>()?;
        let mut in_vk: Vec<&fuel_vulkan_backend::VulkanStorageBytes> = Vec::with_capacity(n_inputs);
        for g in &in_guards {
            in_vk.push(vulkan_input(g)?);
        }
        let backend = in_vk[0].backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::concat::concat_f32: inputs[0] has no VulkanBackend handle. \
                 Storages flowing through the pipelined-executor binding-table dispatch \
                 must come from alloc_bytes_handle / upload_bytes_handle."
                    .to_string(),
            ).bt()
        })?.clone();

        // Helper: construct the layout for an intermediate whose
        // concat-dim size is `cum_dim`. Uses the first input's shape
        // as the template (all inputs share every dim except the
        // concat axis).
        let template_dims: Vec<usize> = layouts[0].shape().dims().to_vec();
        let make_layout = |cum_dim: usize| -> Layout {
            let mut dims = template_dims.clone();
            dims[dim] = cum_dim;
            Layout::contiguous(Shape::from_dims(&dims))
        };

        // Step 0: concat inputs[0] + inputs[1] → tmp0 (or out if N==2).
        let mut acc_dim = input_dim_sizes[0] + input_dim_sizes[1];
        let acc_elems = outer_count * acc_dim * inner_count;
        let mut acc_storage = if n_inputs == 2 {
            // unreachable here — N==2 fast path returned earlier.
            unreachable!()
        } else {
            let bytes = backend.alloc_bytes_handle(acc_elems * 4).map_err(|e| {
                Error::Msg(format!(
                    "vulkan_dispatch::concat::concat_f32: alloc intermediate failed: {e}",
                )).bt()
            })?;
            Storage::new(BackendStorage::Vulkan(bytes), DType::F32)
        };
        {
            let acc_vk = match &mut acc_storage.inner {
                BackendStorage::Vulkan(v) => v,
                _ => unreachable!("just allocated as Vulkan"),
            };
            backend.concat_along_dim_f32_bytes(
                in_vk[0], in_vk[1], acc_vk, dim,
                &layouts[0], &layouts[1],
            )?;
        }

        // Middle steps: concat acc + inputs[i] → new tmp.
        for i in 2..(n_inputs - 1) {
            let new_acc_dim = acc_dim + input_dim_sizes[i];
            let new_elems = outer_count * new_acc_dim * inner_count;
            let new_bytes = backend.alloc_bytes_handle(new_elems * 4).map_err(|e| {
                Error::Msg(format!(
                    "vulkan_dispatch::concat::concat_f32: alloc intermediate {i} failed: {e}",
                )).bt()
            })?;
            let mut new_storage = Storage::new(BackendStorage::Vulkan(new_bytes), DType::F32);

            let acc_layout = make_layout(acc_dim);
            let acc_vk_ref = match &acc_storage.inner {
                BackendStorage::Vulkan(v) => v,
                _ => unreachable!(),
            };
            let new_vk = match &mut new_storage.inner {
                BackendStorage::Vulkan(v) => v,
                _ => unreachable!(),
            };
            backend.concat_along_dim_f32_bytes(
                acc_vk_ref, in_vk[i], new_vk, dim,
                &acc_layout, &layouts[i],
            )?;

            // Drop old acc (releases its buffer back to the VMA pool).
            acc_storage = new_storage;
            acc_dim = new_acc_dim;
        }

        // Final step: concat acc + inputs[N-1] → outputs[0] (the
        // pre-allocated final output).
        let acc_layout = make_layout(acc_dim);
        let acc_vk_ref = match &acc_storage.inner {
            BackendStorage::Vulkan(v) => v,
            _ => unreachable!(),
        };
        let mut out_guard = write_storage(&outputs[0])?;
        let out_vk = vulkan_output(&mut out_guard)?;
        backend.concat_along_dim_f32_bytes(
            acc_vk_ref, in_vk[n_inputs - 1], out_vk, dim,
            &acc_layout, &layouts[n_inputs - 1],
        )
    }
}

// ===========================================================================
// Reduce — Sum / Max / Min f32 (V.2.D)
// ===========================================================================
//
// The Vulkan kernel handles two fast paths only:
//   - full reduction (all dims or empty dims)
//   - single last-dim reduction
// Other dim combinations bail; the route picker can fall back to CPU.
//
// `MeanReduce` is deferred to V.3 — the legacy Vulkan kernel doesn't
// support it (no scalar-divide pass; would need a sum + affine
// follow-up).

macro_rules! vk_reduce_f32_wrapper {
    ($name:ident, $op_id:expr, $label:expr $(,)?) => {
        pub fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::reduce::{}: expected 1 input + 1 output, got {} + {}",
                    $label, inputs.len(), outputs.len(),
                )).bt());
            }
            let dims = match params {
                OpParams::Reduce { dims, keepdim: _ } => dims.clone(),
                other => {
                    return Err(Error::Msg(format!(
                        "vulkan_dispatch::reduce::{}: expected OpParams::Reduce, got {:?}",
                        $label, other,
                    )).bt());
                }
            };
            let layout = layouts.first().ok_or_else(|| {
                Error::Msg(format!(
                    "vulkan_dispatch::reduce::{}: layouts[0] required",
                    $label,
                )).bt()
            })?;
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let a = vulkan_input(&in_guard)?;
            let backend = a.backend().ok_or_else(|| {
                Error::Msg(format!(
                    "vulkan_dispatch::reduce::{}: input has no VulkanBackend handle. \
                     Storages flowing through the pipelined-executor binding-table \
                     dispatch must come from alloc_bytes_handle / upload_bytes_handle.",
                    $label,
                )).bt()
            })?;
            let out = vulkan_output(&mut out_guard)?;
            backend.reduce_f32_bytes($op_id, $label, a, out, layout, &dims)
        }
    };
}

// V.3.G (2026-05-30): non-f32 dtypes only support the last-dim fast
// path. The macro parameterizes the backend method so f16/bf16/f64
// each get their 4-op surface (sum/max/min/mean) without dispatching
// through the f32 unified entry point. For dim combos other than the
// single-last-dim case, the backend method bails and the executor
// falls back to CPU.
// ----- ArgMax / ArgMin along last-dim (V.3.G.arg_reduce, 2026-05-30).
// Output is U32 indices (one per row). Reuses OpParams::Reduce with
// a single-dim constraint matching CUDA. Only last-dim is native;
// other dim combos fall back to CPU.
macro_rules! vk_arg_reduce_wrapper {
    ($name:ident, $op_id:expr, $label:expr, $dtype:expr $(,)?) => {
        pub fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::arg_reduce::{}: expected 1 input + 1 output, got {} + {}",
                    $label, inputs.len(), outputs.len(),
                )).bt());
            }
            let dim = match params {
                OpParams::Reduce { dims, keepdim: _ } => {
                    if dims.len() != 1 {
                        return Err(Error::Msg(format!(
                            "vulkan_dispatch::arg_reduce::{}: expected single reduce dim, got {dims:?}",
                            $label,
                        )).bt());
                    }
                    dims[0]
                }
                other => {
                    return Err(Error::Msg(format!(
                        "vulkan_dispatch::arg_reduce::{}: expected OpParams::Reduce, got {other:?}",
                        $label,
                    )).bt());
                }
            };
            let layout = layouts.first().ok_or_else(|| {
                Error::Msg(format!(
                    "vulkan_dispatch::arg_reduce::{}: layouts empty", $label,
                )).bt()
            })?;
            let shape = layout.shape();
            let rank = shape.dims().len();
            if dim >= rank {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::arg_reduce::{}: dim {dim} >= rank {rank}",
                    $label,
                )).bt());
            }
            let dims_slice = shape.dims();
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let a = vulkan_input(&in_guard)?;
            let backend = a.backend().ok_or_else(|| {
                Error::Msg(format!(
                    "vulkan_dispatch::arg_reduce::{}: input has no VulkanBackend handle.",
                    $label,
                )).bt()
            })?;
            let out = vulkan_output(&mut out_guard)?;
            if dim == rank - 1 {
                let last_dim = dims_slice[rank - 1];
                let outer_count: usize = dims_slice[..rank - 1].iter().product::<usize>().max(1);
                backend.arg_reduce_last_dim_bytes($dtype, $op_id, $label, a, out, outer_count, last_dim)
            } else {
                let n_outer: usize = dims_slice[..dim].iter().product::<usize>().max(1);
                let d_dim = dims_slice[dim];
                let n_inner: usize = dims_slice[dim + 1..].iter().product::<usize>().max(1);
                backend.arg_reduce_any_dim_bytes($dtype, $op_id, $label, a, out, n_outer, d_dim, n_inner)
            }
        }
    };
}

pub mod index_add {
    use super::*;

    pub fn index_add_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        index_add_dispatch(inputs, outputs, params, "index_add_f32",
            |b, base, idx, src, out, oc, bds, ni, ic| {
                b.index_add_f32_bytes(base, idx, src, out, oc, bds, ni, ic)
            })
    }

    pub fn index_add_f64(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        index_add_dispatch(inputs, outputs, params, "index_add_f64",
            |b, base, idx, src, out, oc, bds, ni, ic| {
                b.index_add_f64_bytes(base, idx, src, out, oc, bds, ni, ic)
            })
    }

    pub fn index_add_bf16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        index_add_dispatch(inputs, outputs, params, "index_add_bf16",
            |b, base, idx, src, out, oc, bds, ni, ic| {
                b.index_add_bf16_bytes(base, idx, src, out, oc, bds, ni, ic)
            })
    }

    pub fn index_add_f16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        index_add_dispatch(inputs, outputs, params, "index_add_f16",
            |b, base, idx, src, out, oc, bds, ni, ic| {
                b.index_add_f16_bytes(base, idx, src, out, oc, bds, ni, ic)
            })
    }

    fn index_add_dispatch<F>(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        params: &OpParams,
        debug_name: &'static str,
        f: F,
    ) -> Result<()>
    where
        F: FnOnce(
            &fuel_vulkan_backend::VulkanBackend,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &mut fuel_vulkan_backend::VulkanStorageBytes,
            usize, usize, usize, usize,
        ) -> Result<()>,
    {
        if inputs.len() != 3 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::index_add::{debug_name}: expected 3 inputs + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (outer_count, base_dim_size, n_indices, inner_count) = match params {
            OpParams::IndexAdd { outer_count, base_dim_size, n_indices, inner_count } => {
                (*outer_count, *base_dim_size, *n_indices, *inner_count)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::index_add::{debug_name}: expected OpParams::IndexAdd, got {other:?}",
                )).bt());
            }
        };
        let base_guard = read_storage(&inputs[0])?;
        let idx_guard = read_storage(&inputs[1])?;
        if idx_guard.dtype != DType::U32 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::index_add::{debug_name}: indices must be U32, got {:?}",
                idx_guard.dtype,
            )).bt());
        }
        let src_guard = read_storage(&inputs[2])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let base = vulkan_input(&base_guard)?;
        let indices = vulkan_input(&idx_guard)?;
        let src = vulkan_input(&src_guard)?;
        let backend = base.backend().ok_or_else(|| {
            Error::Msg(format!(
                "vulkan_dispatch::index_add::{debug_name}: base has no VulkanBackend handle.",
            )).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        f(backend, base, indices, src, out, outer_count, base_dim_size, n_indices, inner_count)
    }
}

pub mod scatter_add {
    use super::*;

    /// ScatterAdd along `dim` — f32. 3 inputs (base, U32 indices, src)
    /// → 1 output (same shape as base). Output starts initialized to
    /// base; the kernel atomically accumulates src into the indexed
    /// positions.
    pub fn scatter_add_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 3 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::scatter_add::scatter_add_f32: expected 3 inputs + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (base_shape, src_shape, dim) = match params {
            OpParams::ScatterAdd { base_shape, src_shape, dim } => {
                (base_shape.clone(), src_shape.clone(), *dim)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::scatter_add::scatter_add_f32: expected OpParams::ScatterAdd, got {other:?}",
                )).bt());
            }
        };
        let base_guard = read_storage(&inputs[0])?;
        let idx_guard = read_storage(&inputs[1])?;
        if idx_guard.dtype != DType::U32 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::scatter_add::scatter_add_f32: indices must be U32, got {:?}",
                idx_guard.dtype,
            )).bt());
        }
        let src_guard = read_storage(&inputs[2])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let base = vulkan_input(&base_guard)?;
        let indices = vulkan_input(&idx_guard)?;
        let src = vulkan_input(&src_guard)?;
        let backend = base.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::scatter_add::scatter_add_f32: base has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.scatter_add_f32_bytes(base, indices, src, out, &base_shape, &src_shape, dim)
    }

    /// ScatterAdd along `dim` — f64. Same shape as scatter_add_f32 but
    /// uses u64 CAS for atomic double add. Requires shaderFloat64 +
    /// shaderInt64.
    pub fn scatter_add_f64(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 3 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::scatter_add::scatter_add_f64: expected 3 inputs + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (base_shape, src_shape, dim) = match params {
            OpParams::ScatterAdd { base_shape, src_shape, dim } => {
                (base_shape.clone(), src_shape.clone(), *dim)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::scatter_add::scatter_add_f64: expected OpParams::ScatterAdd, got {other:?}",
                )).bt());
            }
        };
        let base_guard = read_storage(&inputs[0])?;
        let idx_guard = read_storage(&inputs[1])?;
        if idx_guard.dtype != DType::U32 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::scatter_add::scatter_add_f64: indices must be U32, got {:?}",
                idx_guard.dtype,
            )).bt());
        }
        let src_guard = read_storage(&inputs[2])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let base = vulkan_input(&base_guard)?;
        let indices = vulkan_input(&idx_guard)?;
        let src = vulkan_input(&src_guard)?;
        let backend = base.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::scatter_add::scatter_add_f64: base has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.scatter_add_f64_bytes(base, indices, src, out, &base_shape, &src_shape, dim)
    }

    /// ScatterAdd along `dim` — bf16. 2-byte elements; backend runs
    /// a sub-word CAS on packed-u32 output.
    pub fn scatter_add_bf16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        scatter_add_subword(inputs, outputs, params, DType::BF16, "scatter_add_bf16",
            |backend, base, indices, src, out, bs, ss, dim| {
                backend.scatter_add_bf16_bytes(base, indices, src, out, bs, ss, dim)
            })
    }

    /// ScatterAdd along `dim` — f16. Same shape as bf16 path.
    pub fn scatter_add_f16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        scatter_add_subword(inputs, outputs, params, DType::F16, "scatter_add_f16",
            |backend, base, indices, src, out, bs, ss, dim| {
                backend.scatter_add_f16_bytes(base, indices, src, out, bs, ss, dim)
            })
    }

    fn scatter_add_subword<F>(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        params: &OpParams,
        expected_dtype: DType,
        debug_name: &'static str,
        f: F,
    ) -> Result<()>
    where
        F: FnOnce(
            &fuel_vulkan_backend::VulkanBackend,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &mut fuel_vulkan_backend::VulkanStorageBytes,
            &[usize],
            &[usize],
            usize,
        ) -> Result<()>,
    {
        if inputs.len() != 3 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::scatter_add::{debug_name}: expected 3 inputs + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (base_shape, src_shape, dim) = match params {
            OpParams::ScatterAdd { base_shape, src_shape, dim } => {
                (base_shape.clone(), src_shape.clone(), *dim)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::scatter_add::{debug_name}: expected OpParams::ScatterAdd, got {other:?}",
                )).bt());
            }
        };
        let _ = expected_dtype;
        let base_guard = read_storage(&inputs[0])?;
        let idx_guard = read_storage(&inputs[1])?;
        if idx_guard.dtype != DType::U32 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::scatter_add::{debug_name}: indices must be U32, got {:?}",
                idx_guard.dtype,
            )).bt());
        }
        let src_guard = read_storage(&inputs[2])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let base = vulkan_input(&base_guard)?;
        let indices = vulkan_input(&idx_guard)?;
        let src = vulkan_input(&src_guard)?;
        let backend = base.backend().ok_or_else(|| {
            Error::Msg(format!(
                "vulkan_dispatch::scatter_add::{debug_name}: base has no VulkanBackend handle.",
            )).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        f(backend, base, indices, src, out, &base_shape, &src_shape, dim)
    }
}

pub mod arg_reduce {
    use super::*;

    vk_arg_reduce_wrapper!(argmax_f32,  0, "argmax_f32",  DType::F32);
    vk_arg_reduce_wrapper!(argmin_f32,  1, "argmin_f32",  DType::F32);
    vk_arg_reduce_wrapper!(argmax_f16,  0, "argmax_f16",  DType::F16);
    vk_arg_reduce_wrapper!(argmin_f16,  1, "argmin_f16",  DType::F16);
    vk_arg_reduce_wrapper!(argmax_bf16, 0, "argmax_bf16", DType::BF16);
    vk_arg_reduce_wrapper!(argmin_bf16, 1, "argmin_bf16", DType::BF16);
    vk_arg_reduce_wrapper!(argmax_f64,  0, "argmax_f64",  DType::F64);
    vk_arg_reduce_wrapper!(argmin_f64,  1, "argmin_f64",  DType::F64);
}

macro_rules! vk_reduce_last_dim_wrapper {
    ($name:ident, $op_id:expr, $label:expr, $method:ident $(,)?) => {
        pub fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::reduce::{}: expected 1 input + 1 output, got {} + {}",
                    $label, inputs.len(), outputs.len(),
                )).bt());
            }
            let dims = match params {
                OpParams::Reduce { dims, keepdim: _ } => dims.clone(),
                other => {
                    return Err(Error::Msg(format!(
                        "vulkan_dispatch::reduce::{}: expected OpParams::Reduce, got {:?}",
                        $label, other,
                    )).bt());
                }
            };
            let layout = layouts.first().ok_or_else(|| {
                Error::Msg(format!(
                    "vulkan_dispatch::reduce::{}: layouts[0] required",
                    $label,
                )).bt()
            })?;
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let a = vulkan_input(&in_guard)?;
            let backend = a.backend().ok_or_else(|| {
                Error::Msg(format!(
                    "vulkan_dispatch::reduce::{}: input has no VulkanBackend handle.",
                    $label,
                )).bt()
            })?;
            let out = vulkan_output(&mut out_guard)?;
            backend.$method($op_id, $label, a, out, layout, &dims)
        }
    };
}

pub mod reduce {
    use super::*;

    vk_reduce_f32_wrapper!(sum_f32,  0, "reduce_sum_f32");
    vk_reduce_f32_wrapper!(max_f32,  1, "reduce_max_f32");
    vk_reduce_f32_wrapper!(min_f32,  2, "reduce_min_f32");
    // V.3.A.2: Mean added to reduce.slang + reduce_last_dim.slang as
    // op_id=3 (sum then divide by element count).
    vk_reduce_f32_wrapper!(mean_f32, 3, "reduce_mean_f32");

    // ----- V.3.G (2026-05-30): f16/bf16/f64 per-row reductions.
    vk_reduce_last_dim_wrapper!(sum_f16,  0, "reduce_sum_f16",  reduce_f16_bytes);
    vk_reduce_last_dim_wrapper!(max_f16,  1, "reduce_max_f16",  reduce_f16_bytes);
    vk_reduce_last_dim_wrapper!(min_f16,  2, "reduce_min_f16",  reduce_f16_bytes);
    vk_reduce_last_dim_wrapper!(mean_f16, 3, "reduce_mean_f16", reduce_f16_bytes);

    vk_reduce_last_dim_wrapper!(sum_bf16,  0, "reduce_sum_bf16",  reduce_bf16_bytes);
    vk_reduce_last_dim_wrapper!(max_bf16,  1, "reduce_max_bf16",  reduce_bf16_bytes);
    vk_reduce_last_dim_wrapper!(min_bf16,  2, "reduce_min_bf16",  reduce_bf16_bytes);
    vk_reduce_last_dim_wrapper!(mean_bf16, 3, "reduce_mean_bf16", reduce_bf16_bytes);

    vk_reduce_last_dim_wrapper!(sum_f64,  0, "reduce_sum_f64",  reduce_f64_bytes);
    vk_reduce_last_dim_wrapper!(max_f64,  1, "reduce_max_f64",  reduce_f64_bytes);
    vk_reduce_last_dim_wrapper!(min_f64,  2, "reduce_min_f64",  reduce_f64_bytes);
    vk_reduce_last_dim_wrapper!(mean_f64, 3, "reduce_mean_f64", reduce_f64_bytes);
}

// ===========================================================================
// IndexSelect — f32 source + u32 ids (V.2.D)
// ===========================================================================
//
// Two inputs (src, ids) + one output. `OpParams::IndexSelect` carries
// the pre-computed geometry (outer/inner counts, source dim size,
// index count). The Slang kernel is dtype-aware via the descriptor
// binding (f32 storage buffer); V.3 fans out to other source dtypes.

pub mod indexing {
    use super::*;

    pub fn index_select_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::indexing::index_select_f32: expected 2 inputs + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (outer_count, source_dim_size, n_indices, inner_count) = match params {
            OpParams::IndexSelect { outer_count, source_dim_size, n_indices, inner_count } => {
                (*outer_count, *source_dim_size, *n_indices, *inner_count)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::indexing::index_select_f32: expected OpParams::IndexSelect, got {:?}",
                    other,
                )).bt());
            }
        };
        let src_guard = read_storage(&inputs[0])?;
        let ids_guard = read_storage(&inputs[1])?;
        if ids_guard.dtype != DType::U32 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::indexing::index_select_f32: ids must be U32, got {:?}",
                ids_guard.dtype,
            )).bt());
        }
        let mut out_guard = write_storage(&outputs[0])?;
        let src = vulkan_input(&src_guard)?;
        let ids = vulkan_input(&ids_guard)?;
        let backend = src.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::indexing::index_select_f32: src has no VulkanBackend handle. \
                 Storages flowing through the pipelined-executor binding-table dispatch \
                 must come from alloc_bytes_handle / upload_bytes_handle."
                    .to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.index_select_f32_bytes(
            src, ids, out,
            outer_count, source_dim_size, n_indices, inner_count,
        )
    }

    pub fn index_select_f16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        index_select_typed("index_select_f16", inputs, outputs, layouts, params, |b, s, i, o, oc, sd, ni, ic| {
            b.index_select_f16_bytes(s, i, o, oc, sd, ni, ic)
        })
    }

    pub fn index_select_bf16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        index_select_typed("index_select_bf16", inputs, outputs, layouts, params, |b, s, i, o, oc, sd, ni, ic| {
            b.index_select_bf16_bytes(s, i, o, oc, sd, ni, ic)
        })
    }

    pub fn index_select_f64(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        index_select_typed("index_select_f64", inputs, outputs, layouts, params, |b, s, i, o, oc, sd, ni, ic| {
            b.index_select_f64_bytes(s, i, o, oc, sd, ni, ic)
        })
    }

    fn index_select_typed<F>(
        label: &'static str,
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
        call: F,
    ) -> Result<()>
    where
        F: FnOnce(
            &fuel_vulkan_backend::VulkanBackend,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &mut fuel_vulkan_backend::VulkanStorageBytes,
            usize, usize, usize, usize,
        ) -> fuel_core_types::Result<()>,
    {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::indexing::{label}: expected 2 inputs + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (outer_count, source_dim_size, n_indices, inner_count) = match params {
            OpParams::IndexSelect { outer_count, source_dim_size, n_indices, inner_count } => {
                (*outer_count, *source_dim_size, *n_indices, *inner_count)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::indexing::{label}: expected OpParams::IndexSelect, got {other:?}",
                )).bt());
            }
        };
        let src_guard = read_storage(&inputs[0])?;
        let ids_guard = read_storage(&inputs[1])?;
        if ids_guard.dtype != DType::U32 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::indexing::{label}: ids must be U32, got {:?}",
                ids_guard.dtype,
            )).bt());
        }
        let mut out_guard = write_storage(&outputs[0])?;
        let src = vulkan_input(&src_guard)?;
        let ids = vulkan_input(&ids_guard)?;
        let backend = src.backend().ok_or_else(|| {
            Error::Msg(format!(
                "vulkan_dispatch::indexing::{label}: src has no VulkanBackend handle.",
            )).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        call(backend, src, ids, out, outer_count, source_dim_size, n_indices, inner_count)
    }
}

// ===========================================================================
// V.3.E.3+4 — bf16 fan-out via u32-packed manual math
// ===========================================================================

macro_rules! vk_unary_bf16_wrapper {
    ($name:ident, $op_id:expr, $label:expr $(,)?) => {
        pub fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            _params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::unary_bf16::{}: expected 1 input + 1 output, got {} + {}",
                    $label, inputs.len(), outputs.len(),
                )).bt());
            }
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let a = vulkan_input(&in_guard)?;
            let backend = a.backend().ok_or_else(|| {
                Error::Msg(format!(
                    "vulkan_dispatch::unary_bf16::{}: input has no VulkanBackend handle.",
                    $label,
                )).bt()
            })?;
            let out = vulkan_output(&mut out_guard)?;
            backend.unary_bf16_bytes($op_id, $label, a, out)
        }
    };
}

pub mod unary_bf16 {
    use super::*;
    vk_unary_bf16_wrapper!(neg_bf16,     0,  "unary_neg_bf16");
    vk_unary_bf16_wrapper!(sqr_bf16,     1,  "unary_sqr_bf16");
    vk_unary_bf16_wrapper!(sqrt_bf16,    2,  "unary_sqrt_bf16");
    vk_unary_bf16_wrapper!(exp_bf16,     3,  "unary_exp_bf16");
    vk_unary_bf16_wrapper!(log_bf16,     4,  "unary_log_bf16");
    vk_unary_bf16_wrapper!(sin_bf16,     5,  "unary_sin_bf16");
    vk_unary_bf16_wrapper!(cos_bf16,     6,  "unary_cos_bf16");
    vk_unary_bf16_wrapper!(tanh_bf16,    7,  "unary_tanh_bf16");
    vk_unary_bf16_wrapper!(sigmoid_bf16, 8,  "unary_sigmoid_bf16");
    vk_unary_bf16_wrapper!(silu_bf16,    9,  "unary_silu_bf16");
    vk_unary_bf16_wrapper!(gelu_bf16,    10, "unary_gelu_bf16");
    vk_unary_bf16_wrapper!(relu_bf16,    11, "unary_relu_bf16");
    vk_unary_bf16_wrapper!(step_bf16,    12, "unary_step_bf16");
    vk_unary_bf16_wrapper!(abs_bf16,     13, "unary_abs_bf16");
    vk_unary_bf16_wrapper!(sign_bf16,    14, "unary_sign_bf16");
    vk_unary_bf16_wrapper!(recip_bf16,   15, "unary_recip_bf16");
}

macro_rules! vk_binary_bf16_wrapper {
    ($name:ident, $op_id:expr, $label:expr $(,)?) => {
        pub fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            layouts: &[Layout],
            _params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::binary_bf16::{}: expected 2 inputs + 1 output, got {} + {}",
                    $label, inputs.len(), outputs.len(),
                )).bt());
            }
            if layouts.len() < 2 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::binary_bf16::{}: layouts.len() = {} < 2",
                    $label, layouts.len(),
                )).bt());
            }
            let in0_guard = read_storage(&inputs[0])?;
            let in1_guard = read_storage(&inputs[1])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let a = vulkan_input(&in0_guard)?;
            let b = vulkan_input(&in1_guard)?;
            let backend = a.backend().ok_or_else(|| {
                Error::Msg(format!(
                    "vulkan_dispatch::binary_bf16::{}: input[0] has no VulkanBackend handle.",
                    $label,
                )).bt()
            })?;
            let out = vulkan_output(&mut out_guard)?;
            backend.binary_bf16_bytes($op_id, $label, a, b, out, &layouts[0], &layouts[1])
        }
    };
}

pub mod binary_bf16 {
    use super::*;
    vk_binary_bf16_wrapper!(add_bf16,     0, "binary_add_bf16");
    vk_binary_bf16_wrapper!(sub_bf16,     1, "binary_sub_bf16");
    vk_binary_bf16_wrapper!(mul_bf16,     2, "binary_mul_bf16");
    vk_binary_bf16_wrapper!(div_bf16,     3, "binary_div_bf16");
    vk_binary_bf16_wrapper!(maximum_bf16, 4, "binary_maximum_bf16");
    vk_binary_bf16_wrapper!(minimum_bf16, 5, "binary_minimum_bf16");
}

// ===========================================================================
// Triu / Tril / Flip / Roll — byte-width-keyed (b2/b4/b8) via dtype
// ===========================================================================

fn byte_width_for_dtype(label: &str, dt: DType) -> Result<usize> {
    match dt {
        DType::F32 | DType::I32 | DType::U32 => Ok(4),
        DType::F16 | DType::BF16 => Ok(2),
        DType::F64 | DType::I64 => Ok(8),
        other => Err(Error::Msg(format!(
            "{label}: unsupported dtype {other:?} (need 2/4/8-byte elem)"
        )).bt()),
    }
}

macro_rules! vk_triangular_wrapper {
    ($name:ident, $keep_upper:expr, $label:expr $(,)?) => {
        pub fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::triangular::{}: expected 1 input + 1 output, got {} + {}",
                    $label, inputs.len(), outputs.len(),
                )).bt());
            }
            let (batch_count, rows, cols, diagonal) = match params {
                OpParams::Triangular { batch_count, rows, cols, diagonal } => {
                    (*batch_count, *rows, *cols, *diagonal)
                }
                other => {
                    return Err(Error::Msg(format!(
                        "vulkan_dispatch::triangular::{}: expected OpParams::Triangular, got {:?}",
                        $label, other,
                    )).bt());
                }
            };
            let in_guard = read_storage(&inputs[0])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let dt = in_guard.dtype;
            let byte_width = byte_width_for_dtype($label, dt)?;
            let a = vulkan_input(&in_guard)?;
            let backend = a.backend().ok_or_else(|| {
                Error::Msg(format!(
                    "vulkan_dispatch::triangular::{}: input has no VulkanBackend handle.",
                    $label,
                )).bt()
            })?;
            let out = vulkan_output(&mut out_guard)?;
            backend.triangular_bytes(
                byte_width, $keep_upper, a, out,
                batch_count, rows, cols, diagonal,
            )
        }
    };
}

pub mod triangular {
    use super::*;
    vk_triangular_wrapper!(triu, true,  "triu");
    vk_triangular_wrapper!(tril, false, "tril");
}

pub mod flip {
    use super::*;

    pub fn flip(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::flip::flip: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let axis = match params {
            OpParams::Flip { axis, .. } => *axis,
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::flip::flip: expected OpParams::Flip, got {:?}",
                    other,
                )).bt());
            }
        };
        let layout = layouts.first().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::flip::flip: Flip requires an input layout (layouts[0])".to_string(),
            ).bt()
        })?;
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let byte_width = byte_width_for_dtype("flip", in_guard.dtype)?;
        let a = vulkan_input(&in_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::flip::flip: input has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.flip_bytes(byte_width, a, out, layout, axis)
    }
}

pub mod roll {
    use super::*;

    pub fn roll(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::roll::roll: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (shift, axis) = match params {
            OpParams::Roll { shift, axis, .. } => (*shift, *axis),
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::roll::roll: expected OpParams::Roll, got {:?}",
                    other,
                )).bt());
            }
        };
        let layout = layouts.first().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::roll::roll: Roll requires an input layout (layouts[0])".to_string(),
            ).bt()
        })?;
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let byte_width = byte_width_for_dtype("roll", in_guard.dtype)?;
        let a = vulkan_input(&in_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::roll::roll: input has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.roll_bytes(byte_width, a, out, layout, axis, shift)
    }
}

pub mod cumsum {
    use super::*;

    macro_rules! vk_cumsum_wrapper {
        ($name:ident, $backend_fn:ident, $dt:expr, $op_label:expr) => {
            pub fn $name(
                inputs: &[Arc<RwLock<Storage>>],
                outputs: &mut [Arc<RwLock<Storage>>],
                layouts: &[Layout],
                params: &OpParams,
            ) -> Result<()> {
                if inputs.len() != 1 || outputs.len() != 1 {
                    return Err(Error::Msg(format!(
                        "vulkan_dispatch::cumsum::{}: expected 1 input + 1 output, got {} + {}",
                        $op_label, inputs.len(), outputs.len(),
                    )).bt());
                }
                let axis = match params {
                    OpParams::CumSum { axis, .. } => *axis,
                    other => {
                        return Err(Error::Msg(format!(
                            "vulkan_dispatch::cumsum::{}: expected OpParams::CumSum, got {:?}",
                            $op_label, other,
                        )).bt());
                    }
                };
                let layout = layouts.first().ok_or_else(|| {
                    Error::Msg(format!(
                        "vulkan_dispatch::cumsum::{}: CumSum requires an input layout (layouts[0])",
                        $op_label,
                    )).bt()
                })?;
                let in_guard = read_storage(&inputs[0])?;
                let mut out_guard = write_storage(&outputs[0])?;
                if in_guard.dtype != $dt {
                    return Err(Error::Msg(format!(
                        "vulkan_dispatch::cumsum::{}: dtype mismatch (got {:?}, expected {:?})",
                        $op_label, in_guard.dtype, $dt,
                    )).bt());
                }
                let a = vulkan_input(&in_guard)?;
                let backend = a.backend().ok_or_else(|| {
                    Error::Msg(format!(
                        "vulkan_dispatch::cumsum::{}: input has no VulkanBackend handle.",
                        $op_label,
                    )).bt()
                })?;
                let out = vulkan_output(&mut out_guard)?;
                backend.$backend_fn(a, out, layout, axis)
            }
        };
    }

    vk_cumsum_wrapper!(cumsum_f32, cumsum_f32_bytes, DType::F32, "cumsum_f32");
    vk_cumsum_wrapper!(cumsum_f64, cumsum_f64_bytes, DType::F64, "cumsum_f64");
    vk_cumsum_wrapper!(cumsum_f16, cumsum_f16_bytes, DType::F16, "cumsum_f16");
    vk_cumsum_wrapper!(cumsum_bf16, cumsum_bf16_bytes, DType::BF16, "cumsum_bf16");
}

// ===========================================================================
// QMatMul Q4_0 / Q4_K_M / Q8_0 — F32 activation × quant weight → F32 output.
// ===========================================================================
//
// The quantization type is carried in `OpParams::QMatMul.quant_type`; the
// wrapper switches per family to the right Vulkan kernel:
//   - Q4_0  → fused qmatvec_q4_0 (M=1) or matmul_q4_0_tiled (M>1).
//   - Q4_K_M → dequantize_q4_km to f32 scratch, then matmul_f32_bytes.
//   - Q8_0   → dequantize_q8_0  to f32 scratch, then matmul_f32_bytes.
// Other QuantTypes (Q4_1, Q5_*, Q2K/Q3K/Q5K/Q6K, Q8_1) are not yet wired —
// the wrapper returns an error so the route picker falls back to CPU.

pub mod qmatmul {
    use super::*;
    use fuel_graph::QuantType;

    pub fn qmatmul_vk(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::qmatmul: expected 2 inputs + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (quant_type, batch_count, m, n, k) = match params {
            OpParams::QMatMul { quant_type, batch_count, m, n, k } => {
                (*quant_type, *batch_count, *m, *n, *k)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::qmatmul: expected OpParams::QMatMul, got {:?}",
                    other,
                )).bt());
            }
        };
        let a_guard = read_storage(&inputs[0])?;
        let w_guard = read_storage(&inputs[1])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let a = vulkan_input(&a_guard)?;
        let w = vulkan_input(&w_guard)?;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::qmatmul: input has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        match quant_type {
            QuantType::Q4_0   => backend.matmul_q4_0_bytes(a, w, out, batch_count, m, k, n),
            QuantType::Q4_K_M => backend.matmul_q4_km_bytes(a, w, out, batch_count, m, k, n),
            QuantType::Q8_0   => backend.matmul_q8_0_bytes(a, w, out, batch_count, m, k, n),
            other => Err(Error::Msg(format!(
                "vulkan_dispatch::qmatmul: QuantType {other:?} not wired on Vulkan; \
                 add a dequant-then-matmul path or a fused kernel and retry"
            )).bt()),
        }
    }
}

// ===========================================================================
// Conv2D f32 — im2col + matmul, groups=1.
// ===========================================================================

pub mod flash_attn {
    use super::*;

    /// FlashAttention-shape multi-head attention forward, f32 only.
    /// Naive single-pass kernel (NOT tiled). Supports GQA, causal,
    /// scale, alibi; falls through to other backends on window_left /
    /// window_right / softcap.
    pub fn flash_attn_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() < 3 || inputs.len() > 4 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::flash_attn::flash_attn_f32: expected 3-4 inputs (q, k, v, [alibi]) + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (b, hq, hkv, sq, sk, d, scale, causal, wl, wr, softcap) = match params {
            OpParams::FlashAttn {
                b, hq, hkv, sq, sk, d, k_len,
                softmax_scale, causal,
                window_size_left, window_size_right, softcap,
            } => {
                // Vulkan flash v1 reads the full K extent; a runtime
                // k_len (capacity-K, Phase D) isn't supported here. The
                // route picker keeps decode's capacity-K flash on CUDA.
                if *k_len != *sk {
                    return Err(Error::Msg(format!(
                        "vulkan_dispatch::flash_attn: runtime k_len ({}) != K capacity ({}) \
                         not supported on Vulkan v1; route picker should fall back to CPU/CUDA",
                        *k_len, *sk,
                    )).bt());
                }
                (
                    *b, *hq, *hkv, *sq, *sk, *d,
                    *softmax_scale, *causal,
                    *window_size_left, *window_size_right, *softcap,
                )
            },
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::flash_attn::flash_attn_f32: expected OpParams::FlashAttn, got {other:?}",
                )).bt());
            }
        };
        if wl.is_some() || wr.is_some() || softcap.is_some() {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::flash_attn::flash_attn_f32: window_left={wl:?}, window_right={wr:?}, softcap={softcap:?} \
                 not supported on Vulkan v1; route picker should fall back to CPU/CUDA",
            )).bt());
        }

        let q_guard = read_storage(&inputs[0])?;
        let k_guard = read_storage(&inputs[1])?;
        let v_guard = read_storage(&inputs[2])?;
        let alibi_guard = match inputs.get(3) {
            Some(arc) => Some(read_storage(arc)?),
            None => None,
        };
        let mut out_guard = write_storage(&outputs[0])?;
        let q = vulkan_input(&q_guard)?;
        let k = vulkan_input(&k_guard)?;
        let v = vulkan_input(&v_guard)?;
        let alibi = match &alibi_guard {
            Some(g) => Some(vulkan_input(g)?),
            None => None,
        };
        let backend = q.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::flash_attn::flash_attn_f32: q has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.flash_attn_f32_bytes(
            q, k, v, alibi, out,
            b, hq, hkv, sq, sk, d,
            scale, causal,
        )
    }

    /// FlashAttention forward, bf16. Inputs / outputs / alibi all
    /// bf16; math at f32. Same shape contract and constraint set as
    /// the f32 variant.
    pub fn flash_attn_bf16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        flash_attn_half_typed(inputs, outputs, params, DType::BF16, "flash_attn_bf16",
            |backend, q, k, v, a, out, b, hq, hkv, sq, sk, d, scale, causal| {
                backend.flash_attn_bf16_bytes(q, k, v, a, out, b, hq, hkv, sq, sk, d, scale, causal)
            })
    }

    /// FlashAttention forward, f16. Inputs / outputs / alibi all f16.
    pub fn flash_attn_f16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        flash_attn_half_typed(inputs, outputs, params, DType::F16, "flash_attn_f16",
            |backend, q, k, v, a, out, b, hq, hkv, sq, sk, d, scale, causal| {
                backend.flash_attn_f16_bytes(q, k, v, a, out, b, hq, hkv, sq, sk, d, scale, causal)
            })
    }

    fn flash_attn_half_typed<F>(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        params: &OpParams,
        expected_dtype: DType,
        debug_name: &'static str,
        f: F,
    ) -> Result<()>
    where
        F: FnOnce(
            &fuel_vulkan_backend::VulkanBackend,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &fuel_vulkan_backend::VulkanStorageBytes,
            Option<&fuel_vulkan_backend::VulkanStorageBytes>,
            &mut fuel_vulkan_backend::VulkanStorageBytes,
            usize, usize, usize, usize, usize, usize,
            f32, bool,
        ) -> Result<()>,
    {
        if inputs.len() < 3 || inputs.len() > 4 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::flash_attn::{debug_name}: expected 3-4 inputs + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (b, hq, hkv, sq, sk, d, scale, causal, wl, wr, softcap) = match params {
            OpParams::FlashAttn {
                b, hq, hkv, sq, sk, d, k_len,
                softmax_scale, causal,
                window_size_left, window_size_right, softcap,
            } => {
                // Vulkan flash v1 reads the full K extent; a runtime
                // k_len (capacity-K, Phase D) isn't supported here. The
                // route picker keeps decode's capacity-K flash on CUDA.
                if *k_len != *sk {
                    return Err(Error::Msg(format!(
                        "vulkan_dispatch::flash_attn: runtime k_len ({}) != K capacity ({}) \
                         not supported on Vulkan v1; route picker should fall back to CPU/CUDA",
                        *k_len, *sk,
                    )).bt());
                }
                (
                    *b, *hq, *hkv, *sq, *sk, *d,
                    *softmax_scale, *causal,
                    *window_size_left, *window_size_right, *softcap,
                )
            },
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::flash_attn::{debug_name}: expected OpParams::FlashAttn, got {other:?}",
                )).bt());
            }
        };
        if wl.is_some() || wr.is_some() || softcap.is_some() {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::flash_attn::{debug_name}: window/softcap not supported on Vulkan v1",
            )).bt());
        }

        let q_guard = read_storage(&inputs[0])?;
        let k_guard = read_storage(&inputs[1])?;
        let v_guard = read_storage(&inputs[2])?;
        let alibi_guard = match inputs.get(3) {
            Some(arc) => Some(read_storage(arc)?),
            None => None,
        };
        for (name, dt) in [
            ("q", q_guard.dtype), ("k", k_guard.dtype), ("v", v_guard.dtype),
        ] {
            if dt != expected_dtype {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::flash_attn::{debug_name}: {name} must be {expected_dtype:?}, got {dt:?}",
                )).bt());
            }
        }
        if let Some(g) = &alibi_guard {
            if g.dtype != expected_dtype {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::flash_attn::{debug_name}: alibi must be {expected_dtype:?}, got {:?}", g.dtype,
                )).bt());
            }
        }
        let mut out_guard = write_storage(&outputs[0])?;
        let q = vulkan_input(&q_guard)?;
        let k = vulkan_input(&k_guard)?;
        let v = vulkan_input(&v_guard)?;
        let alibi = match &alibi_guard {
            Some(g) => Some(vulkan_input(g)?),
            None => None,
        };
        let backend = q.backend().ok_or_else(|| {
            Error::Msg(format!(
                "vulkan_dispatch::flash_attn::{debug_name}: q has no VulkanBackend handle.",
            )).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        f(backend, q, k, v, alibi, out, b, hq, hkv, sq, sk, d, scale, causal)
    }

    /// FlashAttention backward — dQ, f32. Inputs (q, k, v, dO, [alibi]);
    /// output dQ (same shape as Q). Bails on window/softcap (route
    /// picker falls through to CPU).
    pub fn flash_attn_backward_q_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        flash_attn_backward_f32_typed(inputs, outputs, params, "flash_attn_backward_q_f32",
            |backend, q, k, v, dgrad, a, out, b, hq, hkv, sq, sk, d, scale, causal| {
                backend.flash_attn_backward_q_f32_bytes(q, k, v, dgrad, a, out, b, hq, hkv, sq, sk, d, scale, causal)
            })
    }

    /// FlashAttention backward — dK, f32.
    pub fn flash_attn_backward_k_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        flash_attn_backward_f32_typed(inputs, outputs, params, "flash_attn_backward_k_f32",
            |backend, q, k, v, dgrad, a, out, b, hq, hkv, sq, sk, d, scale, causal| {
                backend.flash_attn_backward_k_f32_bytes(q, k, v, dgrad, a, out, b, hq, hkv, sq, sk, d, scale, causal)
            })
    }

    /// FlashAttention backward — dV, f32.
    pub fn flash_attn_backward_v_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        flash_attn_backward_f32_typed(inputs, outputs, params, "flash_attn_backward_v_f32",
            |backend, q, k, v, dgrad, a, out, b, hq, hkv, sq, sk, d, scale, causal| {
                backend.flash_attn_backward_v_f32_bytes(q, k, v, dgrad, a, out, b, hq, hkv, sq, sk, d, scale, causal)
            })
    }

    /// Shared dispatch shim body for the three FA backward variants.
    /// Validates input counts (4-5: Q, K, V, dO, optional alibi),
    /// extracts shape params, asserts all inputs are F32, then hands
    /// off to the per-variant backend method.
    #[allow(clippy::too_many_arguments)]
    fn flash_attn_backward_f32_typed<F>(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        params: &OpParams,
        debug_name: &'static str,
        f: F,
    ) -> Result<()>
    where
        F: FnOnce(
            &fuel_vulkan_backend::VulkanBackend,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &fuel_vulkan_backend::VulkanStorageBytes,
            &fuel_vulkan_backend::VulkanStorageBytes,
            Option<&fuel_vulkan_backend::VulkanStorageBytes>,
            &mut fuel_vulkan_backend::VulkanStorageBytes,
            usize, usize, usize, usize, usize, usize,
            f32, bool,
        ) -> Result<()>,
    {
        if inputs.len() < 4 || inputs.len() > 5 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::flash_attn::{debug_name}: expected 4-5 inputs (q, k, v, do, [alibi]) + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (b, hq, hkv, sq, sk, d, scale, causal, wl, wr, softcap) = match params {
            OpParams::FlashAttn {
                b, hq, hkv, sq, sk, d, k_len,
                softmax_scale, causal,
                window_size_left, window_size_right, softcap,
            } => {
                // Vulkan flash v1 reads the full K extent; a runtime
                // k_len (capacity-K, Phase D) isn't supported here. The
                // route picker keeps decode's capacity-K flash on CUDA.
                if *k_len != *sk {
                    return Err(Error::Msg(format!(
                        "vulkan_dispatch::flash_attn: runtime k_len ({}) != K capacity ({}) \
                         not supported on Vulkan v1; route picker should fall back to CPU/CUDA",
                        *k_len, *sk,
                    )).bt());
                }
                (
                    *b, *hq, *hkv, *sq, *sk, *d,
                    *softmax_scale, *causal,
                    *window_size_left, *window_size_right, *softcap,
                )
            },
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::flash_attn::{debug_name}: expected OpParams::FlashAttn, got {other:?}",
                )).bt());
            }
        };
        if wl.is_some() || wr.is_some() || softcap.is_some() {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::flash_attn::{debug_name}: window/softcap not supported on Vulkan v1",
            )).bt());
        }

        let q_guard = read_storage(&inputs[0])?;
        let k_guard = read_storage(&inputs[1])?;
        let v_guard = read_storage(&inputs[2])?;
        let do_guard = read_storage(&inputs[3])?;
        let alibi_guard = match inputs.get(4) {
            Some(arc) => Some(read_storage(arc)?),
            None => None,
        };
        for (name, dt) in [
            ("q", q_guard.dtype), ("k", k_guard.dtype), ("v", v_guard.dtype),
            ("do", do_guard.dtype),
        ] {
            if dt != DType::F32 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::flash_attn::{debug_name}: {name} must be F32, got {dt:?}",
                )).bt());
            }
        }
        if let Some(g) = &alibi_guard {
            if g.dtype != DType::F32 {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::flash_attn::{debug_name}: alibi must be F32, got {:?}", g.dtype,
                )).bt());
            }
        }
        let mut out_guard = write_storage(&outputs[0])?;
        let q = vulkan_input(&q_guard)?;
        let k = vulkan_input(&k_guard)?;
        let v = vulkan_input(&v_guard)?;
        let do_storage = vulkan_input(&do_guard)?;
        let alibi = match &alibi_guard {
            Some(g) => Some(vulkan_input(g)?),
            None => None,
        };
        let backend = q.backend().ok_or_else(|| {
            Error::Msg(format!(
                "vulkan_dispatch::flash_attn::{debug_name}: q has no VulkanBackend handle.",
            )).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        f(backend, q, k, v, do_storage, alibi, out, b, hq, hkv, sq, sk, d, scale, causal)
    }
}

pub mod conv2d {
    use super::*;

    pub fn conv2d_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() < 2 || inputs.len() > 3 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::conv2d::conv2d_f32: expected 2-3 inputs + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        if inputs.len() == 3 {
            // Vulkan conv2d doesn't fuse bias yet; route picker should
            // pick a different alternative (CPU/CUDA fused). Bail loudly
            // rather than silently producing wrong output.
            return Err(Error::Msg(
                "vulkan_dispatch::conv2d::conv2d_f32: bias-fused conv2d not supported \
                 on Vulkan yet; bias is a follow-up. Route picker should choose a \
                 fused-conv alternative (CPU/CUDA) when bias is present.".to_string(),
            ).bt());
        }
        let (x_shape, w_shape, stride, padding, dilation, groups) = match params {
            OpParams::Conv2D { x_shape, w_shape, out_shape: _, stride, padding, dilation, groups } => {
                (*x_shape, *w_shape, *stride, *padding, *dilation, *groups)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::conv2d::conv2d_f32: expected OpParams::Conv2D, got {:?}",
                    other,
                )).bt());
            }
        };
        if dilation != (1, 1) {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::conv2d::conv2d_f32: dilation {dilation:?} != (1,1) \
                 not yet supported on Vulkan; route picker should fall back to CPU/CUDA"
            )).bt());
        }
        let in_guard = read_storage(&inputs[0])?;
        let w_guard = read_storage(&inputs[1])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let input = vulkan_input(&in_guard)?;
        let weight = vulkan_input(&w_guard)?;
        let backend = input.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::conv2d::conv2d_f32: input has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.conv2d_f32_bytes(
            input, weight, out,
            x_shape, w_shape, stride, padding, groups,
        )
    }

    /// Conv2D bf16 via im2col_bf16 + cooperative-matrix bf16 matmul.
    /// Same shape contract as the f32 path; activations + weights +
    /// output all bf16. COOP-ONLY shape constraints: c_out % 16 == 0
    /// AND (h_out * w_out) % 16 == 0. Bails on small shapes — route
    /// picker should fall through to f32 conv2d via Cast.
    pub fn conv2d_bf16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() < 2 || inputs.len() > 3 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::conv2d::conv2d_bf16: expected 2-3 inputs + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        if inputs.len() == 3 {
            return Err(Error::Msg(
                "vulkan_dispatch::conv2d::conv2d_bf16: bias-fused conv2d not supported \
                 on Vulkan yet; route picker should pick a fused-conv alternative when \
                 bias is present.".to_string(),
            ).bt());
        }
        let (x_shape, w_shape, stride, padding, dilation, groups) = match params {
            OpParams::Conv2D { x_shape, w_shape, out_shape: _, stride, padding, dilation, groups } => {
                (*x_shape, *w_shape, *stride, *padding, *dilation, *groups)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::conv2d::conv2d_bf16: expected OpParams::Conv2D, got {other:?}",
                )).bt());
            }
        };
        if dilation != (1, 1) {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::conv2d::conv2d_bf16: dilation {dilation:?} != (1,1) \
                 not yet supported on Vulkan; route picker should fall back to CPU/CUDA",
            )).bt());
        }
        let in_guard = read_storage(&inputs[0])?;
        let w_guard = read_storage(&inputs[1])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let input = vulkan_input(&in_guard)?;
        let weight = vulkan_input(&w_guard)?;
        let backend = input.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::conv2d::conv2d_bf16: input has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.conv2d_bf16_bytes(
            input, weight, out,
            x_shape, w_shape, stride, padding, groups,
        )
    }

    /// Conv2D f16 — sibling of `conv2d_bf16`. Reuses the bf16 im2col
    /// shader (2-byte shuffle is dtype-opaque) + matmul_coop_f16_f16_f16.
    pub fn conv2d_f16(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() < 2 || inputs.len() > 3 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::conv2d::conv2d_f16: expected 2-3 inputs + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        if inputs.len() == 3 {
            return Err(Error::Msg(
                "vulkan_dispatch::conv2d::conv2d_f16: bias-fused conv2d not supported \
                 on Vulkan yet".to_string(),
            ).bt());
        }
        let (x_shape, w_shape, stride, padding, dilation, groups) = match params {
            OpParams::Conv2D { x_shape, w_shape, out_shape: _, stride, padding, dilation, groups } => {
                (*x_shape, *w_shape, *stride, *padding, *dilation, *groups)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::conv2d::conv2d_f16: expected OpParams::Conv2D, got {other:?}",
                )).bt());
            }
        };
        if dilation != (1, 1) {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::conv2d::conv2d_f16: dilation {dilation:?} != (1,1) \
                 not yet supported on Vulkan",
            )).bt());
        }
        let in_guard = read_storage(&inputs[0])?;
        let w_guard = read_storage(&inputs[1])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let input = vulkan_input(&in_guard)?;
        let weight = vulkan_input(&w_guard)?;
        let backend = input.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::conv2d::conv2d_f16: input has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.conv2d_f16_bytes(
            input, weight, out,
            x_shape, w_shape, stride, padding, groups,
        )
    }
}

// ===========================================================================
// Cast F8E4M3 ↔ {F32, F16, BF16} — single wrapper dispatches by dtype pair
// ===========================================================================

pub mod cast_f8e4m3 {
    use super::*;

    pub fn cast_f8e4m3(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        _params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::cast_f8e4m3: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let in_guard = read_storage(&inputs[0])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let src_dtype = in_guard.dtype;
        let dst_dtype = out_guard.dtype;
        let src_elem = match src_dtype {
            DType::F32    => 4,
            DType::F16    => 2,
            DType::BF16   => 2,
            DType::F8E4M3 => 1,
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::cast_f8e4m3: unsupported src dtype {other:?}",
                )).bt());
            }
        };
        let a = vulkan_input(&in_guard)?;
        let n_bytes = a.len_bytes();
        if n_bytes % src_elem != 0 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::cast_f8e4m3: input bytes {n_bytes} not a multiple of \
                 src elem size {src_elem} ({src_dtype:?})",
            )).bt());
        }
        let n = n_bytes / src_elem;
        let backend = a.backend().ok_or_else(|| {
            Error::Msg(
                "vulkan_dispatch::cast_f8e4m3: input has no VulkanBackend handle.".to_string(),
            ).bt()
        })?;
        let out = vulkan_output(&mut out_guard)?;
        backend.cast_f8e4m3_bytes(src_dtype, dst_dtype, a, out, n)
    }
}

// ===========================================================================
// Op::Copy D2H — Vulkan → CPU (bridge-retirement Phase 2, post-9c)
// ===========================================================================

/// Dispatch wrapper for `(OpKind::Copy, [T, T], Vulkan)` — D2H from a
/// Vulkan source storage into a freshly-allocated CPU output. The
/// executor allocates the CPU output before this wrapper runs
/// (see [`crate::pipelined::WorkItemKind::Copy`]); the wrapper calls
/// `VulkanBackend::download_bytes` on the input's attached backend
/// handle and copies the result into the CPU storage.
///
/// Dtype-agnostic at the byte level — one wrapper covers every dtype
/// registered at this key.
///
/// Replaces the Vulkan branch of `BackendStorage::read_to_cpu_bytes`
/// (deleted alongside) — the placeholder that
/// [commit 7a95001a](https://github.com/anthropics/fuel/commit/7a95001a)
/// introduced. After this lands, D2H is a graph node the optimizer
/// can see (identity check #1: "more decisions visible to the
/// DAG-level optimizer").
pub fn copy_to_cpu_vulkan(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    _params: &OpParams,
) -> Result<()> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "copy_to_cpu_vulkan: expected 1 input + 1 output, got {} + {}",
            inputs.len(), outputs.len(),
        )).bt());
    }
    let in_guard = read_storage(&inputs[0])?;
    let vk_src = vulkan_input(&in_guard)?;
    let backend = vk_src.backend().ok_or_else(|| {
        Error::Msg(
            "copy_to_cpu_vulkan: Vulkan input has no backend handle. \
             Storages flowing through the pipelined-executor binding-table \
             Copy path must come from VulkanBackend::alloc_bytes_handle / \
             upload_bytes_handle.".to_string()
        ).bt()
    })?;
    let mut out_guard = write_storage(&outputs[0])?;
    match &mut out_guard.inner {
        BackendStorage::Cpu(_) => {
            let bytes = backend.download_bytes(vk_src)?;
            let dst = crate::dispatch::cpu_output(&mut out_guard)?;
            let n = bytes.len().min(dst.len_bytes());
            dst.bytes_mut()[..n].copy_from_slice(&bytes[..n]);
            Ok(())
        }
        BackendStorage::Vulkan(_) => {
            // Same-device copy: one vkCmdCopyBuffer into a fresh
            // standalone buffer (safety copies on destructive ops /
            // residency machinery emit these; before this routing any
            // Vulkan→Vulkan Copy mis-dispatched into cpu_output).
            let copied = backend.slot_copy_to_new_handle(
                vk_src, 0, vk_src.len_bytes(),
            )?;
            let dst = vulkan_output(&mut out_guard)?;
            *dst = copied;
            Ok(())
        }
        #[allow(unreachable_patterns)]
        other => Err(Error::Msg(format!(
            "copy_to_cpu_vulkan: unsupported output substrate {:?} \
             (Vulkan sources copy to CPU or Vulkan outputs only; \
             cross-vendor GPU transfer goes through host staging as \
             two Copy hops)",
            std::mem::discriminant(other),
        )).bt()),
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
    use crate::fused::{
        PrecisionGuarantee, VULKAN_BYTE_LEVEL_PRECISION, VULKAN_CAST_PRECISION,
        VULKAN_FLOAT_POINTWISE_PRECISION, VULKAN_HALF_POINTWISE_PRECISION,
        VULKAN_MATMUL_PRECISION, VULKAN_MATMUL_TENSORCORE_PRECISION,
        VULKAN_QMATMUL_PRECISION, VULKAN_TRANSCENDENTAL_PRECISION,
    };
    use crate::kernel::KernelCaps;
    let vk = BackendId::Vulkan;
    let f32 = DType::F32;
    let u = |t: DType| [t, t];     // (in, out)
    let b = |t: DType| [t, t, t];  // (lhs, rhs, out)

    // Phase 7.6 step 9c follow-up (2026-05-23 + 2026-05-24):
    // - Per-kernel `PrecisionGuarantee` + cost (session 1, shipped).
    // - Per-kernel `KernelCaps::strided_input` (session 2, this commit).
    //   Each Slang kernel was audited to determine whether it walks
    //   per-input strides (and so can consume non-contiguous inputs
    //   like broadcast / transpose / slice views directly) or
    //   requires the executor's auto-Contiguize pass to materialize
    //   contiguous inputs first.
    //
    // **Conventions for this file**:
    // - Stride-aware kernels opt in via
    //   `KernelCaps::strided_input()`. The executor skips
    //   auto-Contiguize for these so lazy views reach the kernel
    //   unmaterialized.
    // - Contiguous-only kernels keep `KernelCaps::empty()`
    //   (the default) AND get a `// strided-input candidate: <reason>`
    //   comment iff the kernel SHAPE would support an arbitrary-
    //   stride extension. Future perf sweeps grep for that marker
    //   to find conversion work.
    // - Contiguous-only kernels whose shape inherently needs flat
    //   layout (subgroup-tree reductions, tiled matmul, im2col,
    //   destination-as-layout WriteSlice, byte-level
    //   Triu/Tril/Flip/Roll) carry a family-level comment but no
    //   candidate marker.
    //
    // Cost functions are bulk-filled at the end via
    // `fill_unset_cost_for_backend(Vulkan, default_cost_for_op_kind)`.

    // ----- Binary f32 (V.2.A) — IEEE-754 pointwise via binary.slang.
    // STRIDE-AWARE: the Slang kernel decomposes the output index into
    // per-dim coordinates and applies per-input strides (with stride=0
    // for broadcast axes). The `binary_f32_bytes` wrapper packs the
    // input layouts' strides into the Params struct; non-contiguous
    // inputs (broadcast, transpose, slice views) reach the kernel
    // unmaterialized. -----
    let strided = KernelCaps::strided_input();
    table.register_with_caps_and_precision(OpKind::AddElementwise,     &b(f32), vk, binary::add_f32,     strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::SubElementwise,     &b(f32), vk, binary::sub_f32,     strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::MulElementwise,     &b(f32), vk, binary::mul_f32,     strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::DivElementwise,     &b(f32), vk, binary::div_f32,     strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::MaximumElementwise, &b(f32), vk, binary::maximum_f32, strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::MinimumElementwise, &b(f32), vk, binary::minimum_f32, strided, VULKAN_FLOAT_POINTWISE_PRECISION);

    // ----- Unary f32 (V.2.B) — split between pointwise + transcendental.
    // STRIDE-AWARE (converted 2026-05-24): unary.slang now mirrors
    // binary.slang's per-dim decomposition + contig fast path; the
    // `unary_f32_bytes` backend method accepts a Layout and packs
    // strides into the Params. -----
    table.register_with_caps_and_precision(OpKind::NegElementwise,     &u(f32), vk, unary::neg_f32,     strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::SqrElementwise,     &u(f32), vk, unary::sqr_f32,     strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::SqrtElementwise,    &u(f32), vk, unary::sqrt_f32,    strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::ReluElementwise,    &u(f32), vk, unary::relu_f32,    strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::StepElementwise,    &u(f32), vk, unary::step_f32,    strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    // Transcendentals via GLSL.std.450 (3-4 ULP per Vulkan spec).
    table.register_with_caps_and_precision(OpKind::ExpElementwise,     &u(f32), vk, unary::exp_f32,     strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::LogElementwise,     &u(f32), vk, unary::log_f32,     strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::SinElementwise,     &u(f32), vk, unary::sin_f32,     strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::CosElementwise,     &u(f32), vk, unary::cos_f32,     strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::TanhElementwise,    &u(f32), vk, unary::tanh_f32,    strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::SigmoidElementwise, &u(f32), vk, unary::sigmoid_f32, strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::SiluElementwise,    &u(f32), vk, unary::silu_f32,    strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::GeluElementwise,    &u(f32), vk, unary::gelu_f32,    strided, VULKAN_TRANSCENDENTAL_PRECISION);
    // V.3.G.unary (2026-05-30): Abs/Sign/Recip — pure arithmetic.
    table.register_with_caps_and_precision(OpKind::AbsElementwise,     &u(f32), vk, unary::abs_f32,     strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::SignElementwise,    &u(f32), vk, unary::sign_f32,    strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::RecipElementwise,   &u(f32), vk, unary::recip_f32,   strided, VULKAN_FLOAT_POINTWISE_PRECISION);

    // ----- Softmax + RmsNorm last-dim (V.2.C, f32) — per-row reductions
    // with subgroup-tree internal accumulation. CONTIGUOUS-ONLY by
    // design: the kernel issues one workgroup per row and reads the
    // row's elements flat; an arbitrary stride on the last-dim axis
    // would break the workgroup-shared-memory reduction. Auto-
    // Contiguize materializes broadcast/transpose views first. -----
    const SOFTMAX_REASON: &str = "fuel-vulkan-backend SoftmaxLastDim: per-row max + exp + sum (subgroup-tree reduction internally); no static ULP / relative / absolute bound — FADD order is scheduler-determined per dispatch.";
    const RMS_NORM_REASON: &str = "fuel-vulkan-backend RmsNormLastDim: per-row x² + sum (subgroup-tree reduction) + sqrt + divide; no static bound — FADD order is scheduler-determined per dispatch.";
    table.register_with_precision(OpKind::SoftmaxLastDim, &u(f32), vk, softmax::softmax_f32, PrecisionGuarantee::none(SOFTMAX_REASON));
    table.register_with_precision(OpKind::RmsNormLastDim, &u(f32), vk, norm::rms_f32,        PrecisionGuarantee::none(RMS_NORM_REASON));

    // ----- SoftmaxLastDimBackward, f32 + f16/bf16/f64 (V.3.G.softmax-bwd,
    // 2026-05-30). 2 inputs (y, g) → 1 output (dx); reuses
    // OpParams::SoftmaxLastDim. Reduction order is scheduler-determined. -----
    const SOFTMAX_BWD_REASON: &str = "fuel-vulkan-backend SoftmaxLastDimBackward: per-row dot(y,g) reduction + per-element y*(g-dot); no static bound \u{2014} FADD order is scheduler-determined per dispatch.";
    {
        let bf16 = DType::BF16;
        let f16  = DType::F16;
        let f64_d = DType::F64;
        // 3 dtypes in binding key: [y_dtype, g_dtype, dx_dtype]
        table.register_with_precision(OpKind::SoftmaxLastDimBackward, &[f32,   f32,   f32],   vk, softmax::softmax_last_dim_backward_f32,  PrecisionGuarantee::none(SOFTMAX_BWD_REASON));
        table.register_with_precision(OpKind::SoftmaxLastDimBackward, &[f16,   f16,   f16],   vk, softmax::softmax_last_dim_backward_f16,  PrecisionGuarantee::none(SOFTMAX_BWD_REASON));
        table.register_with_precision(OpKind::SoftmaxLastDimBackward, &[bf16,  bf16,  bf16],  vk, softmax::softmax_last_dim_backward_bf16, PrecisionGuarantee::none(SOFTMAX_BWD_REASON));
        table.register_with_precision(OpKind::SoftmaxLastDimBackward, &[f64_d, f64_d, f64_d], vk, softmax::softmax_last_dim_backward_f64,  PrecisionGuarantee::none(SOFTMAX_BWD_REASON));
    }

    // ----- Softmax last-dim, f16/bf16/f64 (V.3.G, 2026-05-30).
    // Same mixed-precision pattern as the RmsNorm variants below:
    // f16/bf16 storage with f32 accumulation/exp/sum; f64 native end-
    // to-end. bf16 uses lane-pair (one u32 per lane covers two bf16
    // values); intermediate exp values are stored to the output as
    // bf16, then re-read and rescaled in Phase 3.
    {
        let f16 = DType::F16;
        let bf16 = DType::BF16;
        let f64 = DType::F64;
        table.register_with_precision(OpKind::SoftmaxLastDim, &u(f16),  vk, softmax::softmax_f16,  PrecisionGuarantee::none(SOFTMAX_REASON));
        table.register_with_precision(OpKind::SoftmaxLastDim, &u(bf16), vk, softmax::softmax_bf16, PrecisionGuarantee::none(SOFTMAX_REASON));
        table.register_with_precision(OpKind::SoftmaxLastDim, &u(f64),  vk, softmax::softmax_f64,  PrecisionGuarantee::none(SOFTMAX_REASON));
    }

    // ----- LayerNormLastDimBackward (V.3.G.layer_norm_bwd, 2026-05-30).
    // f32 + f16/bf16/f64. 3 dtypes in binding key: [x, g, dx]. -----
    {
        let f16 = DType::F16;
        let bf16 = DType::BF16;
        let f64_d = DType::F64;
        table.register_with_precision(OpKind::LayerNormLastDimBackward, &[f32,   f32,   f32],   vk, norm::layer_norm_backward_f32,  PrecisionGuarantee::none(RMS_NORM_REASON));
        table.register_with_precision(OpKind::LayerNormLastDimBackward, &[f16,   f16,   f16],   vk, norm::layer_norm_backward_f16,  PrecisionGuarantee::none(RMS_NORM_REASON));
        table.register_with_precision(OpKind::LayerNormLastDimBackward, &[bf16,  bf16,  bf16],  vk, norm::layer_norm_backward_bf16, PrecisionGuarantee::none(RMS_NORM_REASON));
        table.register_with_precision(OpKind::LayerNormLastDimBackward, &[f64_d, f64_d, f64_d], vk, norm::layer_norm_backward_f64,  PrecisionGuarantee::none(RMS_NORM_REASON));
    }

    // ----- LayerNorm last-dim (V.3.G.layer_norm, 2026-05-30). NEW
    // family on Vulkan; CPU + CUDA also have it via separate paths.
    // Two reductions per row (mean + variance), then per-element
    // normalize.
    {
        let f16 = DType::F16;
        let bf16 = DType::BF16;
        let f64_d = DType::F64;
        table.register_with_precision(OpKind::LayerNormLastDim, &u(f32),   vk, norm::layer_norm_f32,  PrecisionGuarantee::none(RMS_NORM_REASON));
        table.register_with_precision(OpKind::LayerNormLastDim, &u(f16),   vk, norm::layer_norm_f16,  PrecisionGuarantee::none(RMS_NORM_REASON));
        table.register_with_precision(OpKind::LayerNormLastDim, &u(bf16),  vk, norm::layer_norm_bf16, PrecisionGuarantee::none(RMS_NORM_REASON));
        table.register_with_precision(OpKind::LayerNormLastDim, &u(f64_d), vk, norm::layer_norm_f64,  PrecisionGuarantee::none(RMS_NORM_REASON));
    }

    // ----- RmsNorm last-dim, f16/bf16/f64 (V.3.G, 2026-05-30).
    // f16/bf16 mirror baracuda's mixed-precision pattern: storage in
    // half precision, accumulation + rsqrt in f32. f64 is native end-
    // to-end (verifies GLSL.std.450 Sqrt on doubles works under
    // shaderFloat64 — NOT OpenCL.std). bf16 uses the lane-pair scheme
    // (each lane processes one u32 = two bf16 lanes) to avoid bf16-
    // pair write races. CONTIGUOUS-ONLY for all three variants. -----
    {
        let f16 = DType::F16;
        let bf16 = DType::BF16;
        let f64 = DType::F64;
        table.register_with_precision(OpKind::RmsNormLastDim, &u(f16),  vk, norm::rms_f16,  PrecisionGuarantee::none(RMS_NORM_REASON));
        table.register_with_precision(OpKind::RmsNormLastDim, &u(bf16), vk, norm::rms_bf16, PrecisionGuarantee::none(RMS_NORM_REASON));
        table.register_with_precision(OpKind::RmsNormLastDim, &u(f64),  vk, norm::rms_f64,  PrecisionGuarantee::none(RMS_NORM_REASON));
    }

    // ----- RoPE (V.2.C, f32) — pointwise rotation with cos/sin tables.
    // STRIDE-AWARE on `x`: rope.slang's Params struct carries
    // `x_s0/x_s1/x_s_seq/x_s_hd` + an `x_contiguous` fast-path flag
    // and decomposes per-thread index into per-dim coordinates. cos/sin
    // are documented as always-contiguous (the wrapper enforces).
    // Binding-shape MUST match the CPU registration `[x, cos, sin, out]`
    // (4 dtypes). The strided_input cap signals "ANY input may be non-
    // contiguous" — for Rope specifically, only x; the wrapper handles
    // forcing cos/sin contiguous through its own path. -----
    table.register_with_caps_and_precision(OpKind::Rope, &[f32, f32, f32, f32], vk, attention::rope_f32, strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    // V.3.G.rope (2026-05-30): f16/f64 RoPE. bf16 deferred (output-
    // packing requires either InterlockedOr or a pair-thread layout).
    {
        let f16 = DType::F16;
        let f64_d = DType::F64;
        table.register_with_caps_and_precision(OpKind::Rope, &[f16,   f16,   f16,   f16],   vk, attention::rope_f16, strided, VULKAN_HALF_POINTWISE_PRECISION);
        table.register_with_caps_and_precision(OpKind::Rope, &[f64_d, f64_d, f64_d, f64_d], vk, attention::rope_f64, strided, VULKAN_FLOAT_POINTWISE_PRECISION);
        let bf16 = DType::BF16;
        table.register_with_caps_and_precision(OpKind::Rope, &[bf16, bf16, bf16, bf16], vk, attention::rope_bf16, strided, VULKAN_HALF_POINTWISE_PRECISION);
    }

    // ----- IndexSelect (V.2.D, f32 src + u32 ids) — pure gather (byte-level).
    // strided-input candidate: index_select.slang flattens input to
    // (outer, axis, inner) and reads via own-shape strides. Could be
    // extended to walk arbitrary input layout strides at gather time
    // (saves an auto-Contiguize when the source is a transpose view).
    // Indices are u32; their layout-strided variant is a follow-up. -----
    let idx_dts = [f32, DType::U32, f32];
    table.register_with_precision(OpKind::IndexSelect, &idx_dts, vk, indexing::index_select_f32, VULKAN_BYTE_LEVEL_PRECISION);
    // V.3.G.index_select (2026-05-30): f16/bf16/f64.
    {
        let f16 = DType::F16;
        let bf16 = DType::BF16;
        let f64_d = DType::F64;
        table.register_with_precision(OpKind::IndexSelect, &[f16,   DType::U32, f16],   vk, indexing::index_select_f16,  VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::IndexSelect, &[bf16,  DType::U32, bf16],  vk, indexing::index_select_bf16, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::IndexSelect, &[f64_d, DType::U32, f64_d], vk, indexing::index_select_f64,  VULKAN_BYTE_LEVEL_PRECISION);
    }

    // ----- Reduce f32 (V.2.D + V.3.A.2) — Sum / Max / Min / Mean.
    // CONTIGUOUS-ONLY by design: reduce.slang does a tree reduction
    // into workgroup-shared memory; the input is read flat. Strided
    // inputs would require either pre-materialization (current
    // behaviour via auto-Contiguize) or a redesigned reduction
    // schedule that's unlikely to outperform the auto-Contiguize +
    // current kernel. -----
    const SUM_REASON: &str = "fuel-vulkan-backend SumReduce: subgroup tree reduction; FADD order depends on workgroup scheduling per dispatch. No static bound applies.";
    const MAX_REASON: &str = "fuel-vulkan-backend MaxReduce: subgroup tree reduction; element selection is deterministic but the comparison order across subgroups depends on scheduling.";
    const MIN_REASON: &str = "fuel-vulkan-backend MinReduce: subgroup tree reduction; same scheduler-dependence as MaxReduce.";
    const MEAN_REASON: &str = "fuel-vulkan-backend MeanReduce: subgroup tree reduction + scalar division; accumulation order is scheduler-determined.";
    table.register_with_precision(OpKind::SumReduce,  &u(f32), vk, reduce::sum_f32,  PrecisionGuarantee::none(SUM_REASON));
    table.register_with_precision(OpKind::MaxReduce,  &u(f32), vk, reduce::max_f32,  PrecisionGuarantee::none(MAX_REASON));
    table.register_with_precision(OpKind::MinReduce,  &u(f32), vk, reduce::min_f32,  PrecisionGuarantee::none(MIN_REASON));
    table.register_with_precision(OpKind::MeanReduce, &u(f32), vk, reduce::mean_f32, PrecisionGuarantee::none(MEAN_REASON));

    // ----- IndexAdd via uint/u64/sub-word CAS (V.3.G.index_add,
    // 2026-05-30). Same wrapper shape as ScatterAdd; output starts
    // initialized to base; the kernel atomically accumulates src
    // at the rank-1 index positions. -----
    {
        const INDEX_ADD_REASON: &str = "fuel-vulkan-backend IndexAdd: atomic float add via uint/u64/sub-word CAS loop; FADD order is scheduler-determined per dispatch.";
        let u32_d  = DType::U32;
        let f64_d  = DType::F64;
        let f16_d  = DType::F16;
        let bf16_d = DType::BF16;
        // 4-dtype binding key: [base, indices, src, out]
        table.register_with_precision(OpKind::IndexAdd, &[f32,    u32_d, f32,    f32   ], vk, index_add::index_add_f32,  PrecisionGuarantee::none(INDEX_ADD_REASON));
        table.register_with_precision(OpKind::IndexAdd, &[f64_d,  u32_d, f64_d,  f64_d ], vk, index_add::index_add_f64,  PrecisionGuarantee::none(INDEX_ADD_REASON));
        table.register_with_precision(OpKind::IndexAdd, &[bf16_d, u32_d, bf16_d, bf16_d], vk, index_add::index_add_bf16, PrecisionGuarantee::none(INDEX_ADD_REASON));
        table.register_with_precision(OpKind::IndexAdd, &[f16_d,  u32_d, f16_d,  f16_d ], vk, index_add::index_add_f16,  PrecisionGuarantee::none(INDEX_ADD_REASON));
    }

    // ----- ScatterAdd along arbitrary dim (V.3.G.scatter_add,
    // 2026-05-30). Output starts initialized to base; the kernel
    // atomically accumulates src via uint CAS (f32) or u64 CAS (f64) —
    // works on stock Vulkan, no VK_EXT_shader_atomic_float required.
    // f16/bf16 would need sub-word CAS and are deferred. -----
    {
        const SCATTER_ADD_REASON: &str = "fuel-vulkan-backend ScatterAdd: atomic float add via uint/u64/sub-word CAS loop; FADD order is scheduler-determined per dispatch.";
        let u32_d = DType::U32;
        let f64_d = DType::F64;
        let f16_d = DType::F16;
        let bf16_d = DType::BF16;
        // 4-dtype binding key: [base, indices, src, out]
        table.register_with_precision(OpKind::ScatterAdd, &[f32,    u32_d, f32,    f32   ], vk, scatter_add::scatter_add_f32,  PrecisionGuarantee::none(SCATTER_ADD_REASON));
        table.register_with_precision(OpKind::ScatterAdd, &[f64_d,  u32_d, f64_d,  f64_d ], vk, scatter_add::scatter_add_f64,  PrecisionGuarantee::none(SCATTER_ADD_REASON));
        table.register_with_precision(OpKind::ScatterAdd, &[bf16_d, u32_d, bf16_d, bf16_d], vk, scatter_add::scatter_add_bf16, PrecisionGuarantee::none(SCATTER_ADD_REASON));
        table.register_with_precision(OpKind::ScatterAdd, &[f16_d,  u32_d, f16_d,  f16_d ], vk, scatter_add::scatter_add_f16,  PrecisionGuarantee::none(SCATTER_ADD_REASON));
    }

    // ----- ArgMaxDim / ArgMinDim along last dim (V.3.G.arg_reduce,
    // 2026-05-30). Output is U32 indices; binding key matches CUDA
    // baracuda registration: [input_dtype, U32]. -----
    {
        let f16 = DType::F16;
        let bf16 = DType::BF16;
        let f64_d = DType::F64;
        let u32_d = DType::U32;
        const ARG_MAX_REASON: &str = "fuel-vulkan-backend ArgMaxDim: tree reduction over (val, idx) pairs; lower index wins on ties — deterministic given input values.";
        const ARG_MIN_REASON: &str = "fuel-vulkan-backend ArgMinDim: same as ArgMaxDim with min comparator.";
        table.register_with_precision(OpKind::ArgMaxDim, &[f32,   u32_d], vk, arg_reduce::argmax_f32,  PrecisionGuarantee::none(ARG_MAX_REASON));
        table.register_with_precision(OpKind::ArgMaxDim, &[f16,   u32_d], vk, arg_reduce::argmax_f16,  PrecisionGuarantee::none(ARG_MAX_REASON));
        table.register_with_precision(OpKind::ArgMaxDim, &[bf16,  u32_d], vk, arg_reduce::argmax_bf16, PrecisionGuarantee::none(ARG_MAX_REASON));
        table.register_with_precision(OpKind::ArgMaxDim, &[f64_d, u32_d], vk, arg_reduce::argmax_f64,  PrecisionGuarantee::none(ARG_MAX_REASON));
        table.register_with_precision(OpKind::ArgMinDim, &[f32,   u32_d], vk, arg_reduce::argmin_f32,  PrecisionGuarantee::none(ARG_MIN_REASON));
        table.register_with_precision(OpKind::ArgMinDim, &[f16,   u32_d], vk, arg_reduce::argmin_f16,  PrecisionGuarantee::none(ARG_MIN_REASON));
        table.register_with_precision(OpKind::ArgMinDim, &[bf16,  u32_d], vk, arg_reduce::argmin_bf16, PrecisionGuarantee::none(ARG_MIN_REASON));
        table.register_with_precision(OpKind::ArgMinDim, &[f64_d, u32_d], vk, arg_reduce::argmin_f64,  PrecisionGuarantee::none(ARG_MIN_REASON));
    }

    // ----- Reduce last-dim, f16/bf16/f64 (V.3.G, 2026-05-30). Only
    // the single-last-dim fast path is native on these dtypes; other
    // dim combos bail and the executor falls back to CPU. -----
    {
        let f16 = DType::F16;
        let bf16 = DType::BF16;
        let f64 = DType::F64;
        table.register_with_precision(OpKind::SumReduce,  &u(f16),  vk, reduce::sum_f16,   PrecisionGuarantee::none(SUM_REASON));
        table.register_with_precision(OpKind::MaxReduce,  &u(f16),  vk, reduce::max_f16,   PrecisionGuarantee::none(MAX_REASON));
        table.register_with_precision(OpKind::MinReduce,  &u(f16),  vk, reduce::min_f16,   PrecisionGuarantee::none(MIN_REASON));
        table.register_with_precision(OpKind::MeanReduce, &u(f16),  vk, reduce::mean_f16,  PrecisionGuarantee::none(MEAN_REASON));
        table.register_with_precision(OpKind::SumReduce,  &u(bf16), vk, reduce::sum_bf16,  PrecisionGuarantee::none(SUM_REASON));
        table.register_with_precision(OpKind::MaxReduce,  &u(bf16), vk, reduce::max_bf16,  PrecisionGuarantee::none(MAX_REASON));
        table.register_with_precision(OpKind::MinReduce,  &u(bf16), vk, reduce::min_bf16,  PrecisionGuarantee::none(MIN_REASON));
        table.register_with_precision(OpKind::MeanReduce, &u(bf16), vk, reduce::mean_bf16, PrecisionGuarantee::none(MEAN_REASON));
        table.register_with_precision(OpKind::SumReduce,  &u(f64),  vk, reduce::sum_f64,   PrecisionGuarantee::none(SUM_REASON));
        table.register_with_precision(OpKind::MaxReduce,  &u(f64),  vk, reduce::max_f64,   PrecisionGuarantee::none(MAX_REASON));
        table.register_with_precision(OpKind::MinReduce,  &u(f64),  vk, reduce::min_f64,   PrecisionGuarantee::none(MIN_REASON));
        table.register_with_precision(OpKind::MeanReduce, &u(f64),  vk, reduce::mean_f64,  PrecisionGuarantee::none(MEAN_REASON));
    }

    // ----- Concat f32 (V.2.D) — memcpy (byte-level).
    // STRIDE-AWARE: concat_along_dim.slang explicitly supports per-
    // operand stride support so either input may be a lazy view
    // (permute, broadcast). N==2 only; N>2 falls back to next alt. -----
    table.register_with_caps_and_precision(OpKind::Concat, &u(f32), vk, concat::concat_f32, strided, VULKAN_BYTE_LEVEL_PRECISION);

    // ----- Pad with constant fill (V.3.G.pad, 2026-05-30).
    // Byte-level kernels (b1/b2/b4/b8) dispatched by the OUTPUT dtype's
    // size at the shim. Reflect/replicate modes fall through to CPU.
    {
        let f16 = DType::F16;
        let bf16 = DType::BF16;
        let f64_d = DType::F64;
        let u8_d = DType::U8;
        let u32_d = DType::U32;
        table.register_with_precision(OpKind::Pad, &u(f32),   vk, pad::pad_const, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::Pad, &u(f16),   vk, pad::pad_const, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::Pad, &u(bf16),  vk, pad::pad_const, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::Pad, &u(f64_d), vk, pad::pad_const, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::Pad, &u(u8_d),  vk, pad::pad_const, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::Pad, &u(u32_d), vk, pad::pad_const, VULKAN_BYTE_LEVEL_PRECISION);

        // ----- PadBackward (V.3.G.pad_backward, 2026-05-30).
        // Const mode only on Vulkan; reflect/replicate fall through
        // to CPU (need atomic float add).
        table.register_with_precision(OpKind::PadBackward, &u(f32),   vk, pad::pad_backward, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::PadBackward, &u(f16),   vk, pad::pad_backward, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::PadBackward, &u(bf16),  vk, pad::pad_backward, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::PadBackward, &u(f64_d), vk, pad::pad_backward, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::PadBackward, &u(u8_d),  vk, pad::pad_backward, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::PadBackward, &u(u32_d), vk, pad::pad_backward, VULKAN_BYTE_LEVEL_PRECISION);

        // ----- MaskedFill (V.3.G.masked_fill, 2026-05-30).
        // 2 inputs (data + U8 mask) → 1 output. Same byte-width family
        // as Pad; one dispatch shim across all dtypes.
        table.register_with_precision(OpKind::MaskedFill, &[f32,   DType::U8, f32],   vk, masked_fill::masked_fill, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::MaskedFill, &[f16,   DType::U8, f16],   vk, masked_fill::masked_fill, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::MaskedFill, &[bf16,  DType::U8, bf16],  vk, masked_fill::masked_fill, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::MaskedFill, &[f64_d, DType::U8, f64_d], vk, masked_fill::masked_fill, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::MaskedFill, &[u8_d,  DType::U8, u8_d],  vk, masked_fill::masked_fill, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::MaskedFill, &[u32_d, DType::U8, u32_d], vk, masked_fill::masked_fill, VULKAN_BYTE_LEVEL_PRECISION);

        // ----- Gather (V.3.G.gather, 2026-05-30).
        // 2 inputs (src + U32 indices) → 1 output. Same byte-width
        // family as Pad/MaskedFill; one dispatch shim across all
        // dtypes.
        table.register_with_precision(OpKind::Gather, &[f32,   DType::U32, f32],   vk, gather::gather, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::Gather, &[f16,   DType::U32, f16],   vk, gather::gather, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::Gather, &[bf16,  DType::U32, bf16],  vk, gather::gather, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::Gather, &[f64_d, DType::U32, f64_d], vk, gather::gather, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::Gather, &[u8_d,  DType::U32, u8_d],  vk, gather::gather, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::Gather, &[u32_d, DType::U32, u32_d], vk, gather::gather, VULKAN_BYTE_LEVEL_PRECISION);
    }
    // V.3.G.concat (2026-05-30): f16/bf16/f64 concat (pure data movement).
    // bf16 uses single-thread-per-bf16 with InterlockedOr half-word
    // writes + zero-fill so adjacent threads writing the same u32 at
    // an odd (a, b) boundary don't race.
    {
        let f16 = DType::F16;
        let bf16 = DType::BF16;
        let f64_d = DType::F64;
        table.register_with_caps_and_precision(OpKind::Concat, &u(f16),   vk, concat::concat_f16,  strided, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_caps_and_precision(OpKind::Concat, &u(bf16),  vk, concat::concat_bf16, strided, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_caps_and_precision(OpKind::Concat, &u(f64_d), vk, concat::concat_f64,  strided, VULKAN_BYTE_LEVEL_PRECISION);
    }

    // ----- MatMul f32 (V.2.D) — deterministic per-output-element FMA
    // accumulation. CONTIGUOUS-ONLY: tiled / reg-tile / matvec
    // kernels load via vec4 / cooperative loads that require
    // contiguous row-major layout; strided inputs go through
    // auto-Contiguize. -----
    table.register_with_precision(OpKind::MatMul, &b(f32), vk, matmul::matmul_f32, VULKAN_MATMUL_PRECISION);

    // ----- MatMul mixed f32 × bf16 → f32 (V.3.D) — tensor cores via
    // matvec_bf16_b / matmul_tiled_bf16_b / matmul_coop. Wider ULP than
    // pure f32 matmul because bf16 inputs lose 16 mantissa bits.
    // CONTIGUOUS-ONLY: tensor-core cooperative-matrix kernels
    // require canonical row-major tile layout. -----
    let bf16 = DType::BF16;
    table.register_with_precision(OpKind::MatMul, &[f32, bf16, f32], vk, matmul::matmul_f32_bf16_b, VULKAN_MATMUL_TENSORCORE_PRECISION);
    // ----- MatMul pure bf16 × bf16 → f32 (V.3 coop-matrix). Both
    // operands stored as bf16, downcast bf16→f16 on shared-mem load,
    // f32 accumulator. COOP-ONLY (M%16==0, N%16==0, K>=16); route
    // picker falls through to cast+f32-matmul on small shapes. -----
    table.register_with_precision(OpKind::MatMul, &[bf16, bf16, f32], vk, matmul::matmul_bf16_bf16_f32, VULKAN_MATMUL_TENSORCORE_PRECISION);
    // ----- MatMul pure bf16 × bf16 → bf16 (downcast store). Same
    // coop[3] tile, but the f32 accumulator is staged to shared mem
    // and packed to bf16 lanes on store. Closes the end-to-end bf16
    // inference chain (next layer can consume bf16 activations). -----
    table.register_with_precision(OpKind::MatMul, &[bf16, bf16, bf16], vk, matmul::matmul_bf16_bf16_bf16, VULKAN_MATMUL_TENSORCORE_PRECISION);
    // ----- MatMul pure f16 × f16 → f16 (downcast store). Sibling
    // of bf16→bf16; native float16_t inputs + `float16BitsToUint16`
    // on the pack step. Closes the f16 inference chain. -----
    {
        let f16_d = DType::F16;
        table.register_with_precision(OpKind::MatMul, &[f16_d, f16_d, f16_d], vk, matmul::matmul_f16_f16_f16, VULKAN_MATMUL_TENSORCORE_PRECISION);
    }
    // ----- MatMul pure f16 × f16 → f32 (V.3 coop-matrix). Native
    // float16_t inputs (no downcast); same coop[3] tile + f32
    // accumulator. Same shape constraints as the bf16 variant. -----
    {
        let f16_d = DType::F16;
        table.register_with_precision(OpKind::MatMul, &[f16_d, f16_d, f32], vk, matmul::matmul_f16_f16_f32, VULKAN_MATMUL_TENSORCORE_PRECISION);
    }

    // ----- Affine f32 (V.2.E) — y = mul*x + add. Pointwise FMA.
    // STRIDE-AWARE (converted 2026-05-24): affine.slang now carries
    // the same per-dim shape/stride + flags Params as unary.slang.
    // `affine_f32_bytes` packs the layout identically to the unary
    // path. -----
    table.register_with_caps_and_precision(OpKind::Affine, &u(f32), vk, affine::affine_f32, strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    {
        let f16_d  = DType::F16;
        let bf16_d = DType::BF16;
        let f64_d  = DType::F64;
        // f64 and f16: stride-aware via native typed buffers.
        table.register_with_caps_and_precision(OpKind::Affine, &u(f64_d), vk, affine::affine_f64, strided, VULKAN_FLOAT_POINTWISE_PRECISION);
        table.register_with_caps_and_precision(OpKind::Affine, &u(f16_d), vk, affine::affine_f16, strided, VULKAN_HALF_POINTWISE_PRECISION);
        // bf16: pair-thread packed-u32 kernel; CONTIGUOUS-ONLY (no strided caps,
        // so Fuel auto-Contiguizes upstream).
        table.register_with_precision(OpKind::Affine, &u(bf16_d), vk, affine::affine_bf16, VULKAN_HALF_POINTWISE_PRECISION);
    }

    // ----- Clamp f32 (V.3.A.1) — pointwise min/max with constants.
    // STRIDE-AWARE (converted 2026-05-24): clamp.slang mirrors the
    // affine.slang stride-aware Params shape. -----
    table.register_with_caps_and_precision(OpKind::ClampElementwise, &u(f32), vk, clamp::clamp_f32, strided, VULKAN_FLOAT_POINTWISE_PRECISION);

    // ----- PowI f32 (V.3.A.3) — repeated FMA, bit-stable on same hardware.
    // STRIDE-AWARE (converted 2026-05-24): powi.slang mirrors the
    // affine.slang stride-aware Params shape. -----
    table.register_with_caps_and_precision(OpKind::PowIElementwise, &u(f32), vk, powi::powi_f32, strided, VULKAN_FLOAT_POINTWISE_PRECISION);

    // ----- Cast (V.3.B) — f32↔f16, f32↔bf16 (pure dtype conversion).
    // strided-input candidate (with caveat): cast kernels pack pairs
    // of f16/bf16 per u32 for storage efficiency, so the input/output
    // must be aligned to even element counts; that constraint
    // composes awkwardly with arbitrary strides. A strided extension
    // is plausible but requires careful handling of the packing
    // boundary. -----
    let f16 = DType::F16;
    let bf16_d = DType::BF16;
    table.register_with_precision(OpKind::Cast, &[f32,    f16],    vk, cast::cast_f32_half, VULKAN_CAST_PRECISION);
    table.register_with_precision(OpKind::Cast, &[f16,    f32],    vk, cast::cast_f32_half, VULKAN_CAST_PRECISION);
    table.register_with_precision(OpKind::Cast, &[f32,    bf16_d], vk, cast::cast_f32_half, VULKAN_CAST_PRECISION);
    table.register_with_precision(OpKind::Cast, &[bf16_d, f32],    vk, cast::cast_f32_half, VULKAN_CAST_PRECISION);
    // V.3.G.cast (2026-05-30): f32↔f64 — the f64 escape hatch for
    // ops that can't run natively on Vulkan f64 (per the GLSL.std.450
    // transcendental precision constraint, the graph-level pattern is
    // Cast→f32→op→Cast).
    {
        let f64_d = DType::F64;
        table.register_with_precision(OpKind::Cast, &[f32,   f64_d], vk, cast::cast_f32_f64, VULKAN_CAST_PRECISION);
        table.register_with_precision(OpKind::Cast, &[f64_d, f32],   vk, cast::cast_f32_f64, VULKAN_CAST_PRECISION);
    }

    // ----- Binary f16 (V.3.E) — native float16_t via shaderFloat16.
    // STRIDE-AWARE: binary_f16.slang mirrors binary.slang's stride-
    // aware Params layout exactly (only the buffer element type differs).
    // The `binary_f16_bytes` wrapper delegates to `binary_typed_bytes`
    // which packs strides identically to the f32 variant. -----
    table.register_with_caps_and_precision(OpKind::AddElementwise,     &b(f16), vk, binary_f16::add_f16,     strided, VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::SubElementwise,     &b(f16), vk, binary_f16::sub_f16,     strided, VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::MulElementwise,     &b(f16), vk, binary_f16::mul_f16,     strided, VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::DivElementwise,     &b(f16), vk, binary_f16::div_f16,     strided, VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::MaximumElementwise, &b(f16), vk, binary_f16::maximum_f16, strided, VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::MinimumElementwise, &b(f16), vk, binary_f16::minimum_f16, strided, VULKAN_HALF_POINTWISE_PRECISION);

    // ----- Unary f16 (V.3.E) — split between half-pointwise + half-transcendental.
    // STRIDE-AWARE (converted 2026-05-24): unary_f16.slang now mirrors
    // binary.slang's stride-aware Params layout. The `unary_f16_bytes`
    // wrapper delegates to `unary_typed_bytes` which packs strides
    // identically to the f32 variant. -----
    table.register_with_caps_and_precision(OpKind::NegElementwise,     &u(f16), vk, unary_f16::neg_f16,     strided, VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::SqrElementwise,     &u(f16), vk, unary_f16::sqr_f16,     strided, VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::SqrtElementwise,    &u(f16), vk, unary_f16::sqrt_f16,    strided, VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::ReluElementwise,    &u(f16), vk, unary_f16::relu_f16,    strided, VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::StepElementwise,    &u(f16), vk, unary_f16::step_f16,    strided, VULKAN_HALF_POINTWISE_PRECISION);
    // f16 transcendentals — same Vulkan-spec 4-ULP envelope, just at half precision.
    table.register_with_caps_and_precision(OpKind::ExpElementwise,     &u(f16), vk, unary_f16::exp_f16,     strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::LogElementwise,     &u(f16), vk, unary_f16::log_f16,     strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::SinElementwise,     &u(f16), vk, unary_f16::sin_f16,     strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::CosElementwise,     &u(f16), vk, unary_f16::cos_f16,     strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::TanhElementwise,    &u(f16), vk, unary_f16::tanh_f16,    strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::SigmoidElementwise, &u(f16), vk, unary_f16::sigmoid_f16, strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::SiluElementwise,    &u(f16), vk, unary_f16::silu_f16,    strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::GeluElementwise,    &u(f16), vk, unary_f16::gelu_f16,    strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::AbsElementwise,     &u(f16), vk, unary_f16::abs_f16,     strided, VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::SignElementwise,    &u(f16), vk, unary_f16::sign_f16,    strided, VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::RecipElementwise,   &u(f16), vk, unary_f16::recip_f16,   strided, VULKAN_HALF_POINTWISE_PRECISION);

    // ----- Binary f64 (V.3.E.5) — native `double` via shaderFloat64.
    // STRIDE-AWARE: binary_f64.slang mirrors binary.slang's stride-aware
    // Params layout exactly. -----
    let f64_d = DType::F64;
    table.register_with_caps_and_precision(OpKind::AddElementwise,     &b(f64_d), vk, binary_f64::add_f64,     strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::SubElementwise,     &b(f64_d), vk, binary_f64::sub_f64,     strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::MulElementwise,     &b(f64_d), vk, binary_f64::mul_f64,     strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::DivElementwise,     &b(f64_d), vk, binary_f64::div_f64,     strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::MaximumElementwise, &b(f64_d), vk, binary_f64::maximum_f64, strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::MinimumElementwise, &b(f64_d), vk, binary_f64::minimum_f64, strided, VULKAN_FLOAT_POINTWISE_PRECISION);

    // ----- Unary f64 (V.3.E.5) — full 13-op surface. Transcendentals
    // implemented via Horner-polynomial approximations (~1e-12 relative
    // precision target) in unary_f64.slang; portable across any
    // shaderFloat64 driver.
    // STRIDE-AWARE (converted 2026-05-24): same Params layout as the
    // f32/f16 variants; `unary_f64_bytes` forwards to `unary_typed_bytes`. -----
    table.register_with_caps_and_precision(OpKind::NegElementwise,     &u(f64_d), vk, unary_f64::neg_f64,     strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::SqrElementwise,     &u(f64_d), vk, unary_f64::sqr_f64,     strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::SqrtElementwise,    &u(f64_d), vk, unary_f64::sqrt_f64,    strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::ReluElementwise,    &u(f64_d), vk, unary_f64::relu_f64,    strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::StepElementwise,    &u(f64_d), vk, unary_f64::step_f64,    strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::ExpElementwise,     &u(f64_d), vk, unary_f64::exp_f64,     strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::LogElementwise,     &u(f64_d), vk, unary_f64::log_f64,     strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::SinElementwise,     &u(f64_d), vk, unary_f64::sin_f64,     strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::AbsElementwise,     &u(f64_d), vk, unary_f64::abs_f64,     strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::SignElementwise,    &u(f64_d), vk, unary_f64::sign_f64,    strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::RecipElementwise,   &u(f64_d), vk, unary_f64::recip_f64,   strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::CosElementwise,     &u(f64_d), vk, unary_f64::cos_f64,     strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::TanhElementwise,    &u(f64_d), vk, unary_f64::tanh_f64,    strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::SigmoidElementwise, &u(f64_d), vk, unary_f64::sigmoid_f64, strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::SiluElementwise,    &u(f64_d), vk, unary_f64::silu_f64,    strided, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_caps_and_precision(OpKind::GeluElementwise,    &u(f64_d), vk, unary_f64::gelu_f64,    strided, VULKAN_TRANSCENDENTAL_PRECISION);

    // ----- WriteSlice (V.3.J) — byte-width-keyed (b1/b2/b4/b8). Pure
    // byte-level data movement; no FP math.
    // CONTIGUOUS-ONLY: write_slice_b*.slang reads `src` contiguously
    // in its own rank-N shape and writes the matching slab inside
    // `dst` (also contiguous in its larger rank-N shape). The
    // destination's slab geometry IS the layout — strided inputs
    // wouldn't compose with the slab-walk's own-shape strides.
    // Auto-Contiguize materializes any non-contiguous source. -----
    table.register_with_precision(OpKind::WriteSlice, &u(f32),         vk, write_slice::write_slice_b4, VULKAN_BYTE_LEVEL_PRECISION);
    table.register_with_precision(OpKind::WriteSlice, &u(DType::I32),  vk, write_slice::write_slice_b4, VULKAN_BYTE_LEVEL_PRECISION);
    table.register_with_precision(OpKind::WriteSlice, &u(DType::U32),  vk, write_slice::write_slice_b4, VULKAN_BYTE_LEVEL_PRECISION);
    table.register_with_precision(OpKind::WriteSlice, &u(f16),         vk, write_slice::write_slice_b2, VULKAN_BYTE_LEVEL_PRECISION);
    table.register_with_precision(OpKind::WriteSlice, &u(bf16_d),      vk, write_slice::write_slice_b2, VULKAN_BYTE_LEVEL_PRECISION);
    table.register_with_precision(OpKind::WriteSlice, &u(DType::F64),  vk, write_slice::write_slice_b8, VULKAN_BYTE_LEVEL_PRECISION);
    table.register_with_precision(OpKind::WriteSlice, &u(DType::I64),  vk, write_slice::write_slice_b8, VULKAN_BYTE_LEVEL_PRECISION);
    table.register_with_precision(OpKind::WriteSlice, &u(DType::U8),   vk, write_slice::write_slice_b1, VULKAN_BYTE_LEVEL_PRECISION);
    table.register_with_precision(OpKind::WriteSlice, &u(DType::I8),   vk, write_slice::write_slice_b1, VULKAN_BYTE_LEVEL_PRECISION);

    // ----- WriteSliceRotating (Phase C) — sliding-window KV writes.
    // Same byte-width-dispatched surface as WriteSlice; the wrapper
    // handles position D2H + ring-boundary split before delegating
    // to baracuda's write_slice_bytes path through VulkanBackend.
    table.register_with_precision(OpKind::WriteSliceRotating, &u(f32),         vk, write_slice_rotating::write_slice_rotating_b4, VULKAN_BYTE_LEVEL_PRECISION);
    table.register_with_precision(OpKind::WriteSliceRotating, &u(DType::I32),  vk, write_slice_rotating::write_slice_rotating_b4, VULKAN_BYTE_LEVEL_PRECISION);
    table.register_with_precision(OpKind::WriteSliceRotating, &u(DType::U32),  vk, write_slice_rotating::write_slice_rotating_b4, VULKAN_BYTE_LEVEL_PRECISION);
    table.register_with_precision(OpKind::WriteSliceRotating, &u(f16),         vk, write_slice_rotating::write_slice_rotating_b2, VULKAN_BYTE_LEVEL_PRECISION);
    table.register_with_precision(OpKind::WriteSliceRotating, &u(bf16_d),      vk, write_slice_rotating::write_slice_rotating_b2, VULKAN_BYTE_LEVEL_PRECISION);
    table.register_with_precision(OpKind::WriteSliceRotating, &u(DType::F64),  vk, write_slice_rotating::write_slice_rotating_b8, VULKAN_BYTE_LEVEL_PRECISION);
    table.register_with_precision(OpKind::WriteSliceRotating, &u(DType::I64),  vk, write_slice_rotating::write_slice_rotating_b8, VULKAN_BYTE_LEVEL_PRECISION);
    table.register_with_precision(OpKind::WriteSliceRotating, &u(DType::U8),   vk, write_slice_rotating::write_slice_rotating_b1, VULKAN_BYTE_LEVEL_PRECISION);
    table.register_with_precision(OpKind::WriteSliceRotating, &u(DType::I8),   vk, write_slice_rotating::write_slice_rotating_b1, VULKAN_BYTE_LEVEL_PRECISION);

    // ----- Binary bf16 (V.3.E.3+4) — u32-packed math via Slang.
    // STRIDE-AWARE: binary_bf16.slang mirrors binary.slang's
    // stride-aware Params layout exactly; only the bf16↔f32
    // round-trip on read/write differs. -----
    table.register_with_caps_and_precision(OpKind::AddElementwise,     &b(bf16_d), vk, binary_bf16::add_bf16,     strided, VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::SubElementwise,     &b(bf16_d), vk, binary_bf16::sub_bf16,     strided, VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::MulElementwise,     &b(bf16_d), vk, binary_bf16::mul_bf16,     strided, VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::DivElementwise,     &b(bf16_d), vk, binary_bf16::div_bf16,     strided, VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::MaximumElementwise, &b(bf16_d), vk, binary_bf16::maximum_bf16, strided, VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::MinimumElementwise, &b(bf16_d), vk, binary_bf16::minimum_bf16, strided, VULKAN_HALF_POINTWISE_PRECISION);

    // ----- Unary bf16 (V.3.E.3+4) — full 13-op surface via u32 packing.
    // strided-input candidate: same shape as unary f32. -----
    table.register_with_precision(OpKind::NegElementwise,     &u(bf16_d), vk, unary_bf16::neg_bf16,     VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_precision(OpKind::SqrElementwise,     &u(bf16_d), vk, unary_bf16::sqr_bf16,     VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_precision(OpKind::SqrtElementwise,    &u(bf16_d), vk, unary_bf16::sqrt_bf16,    VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_precision(OpKind::ReluElementwise,    &u(bf16_d), vk, unary_bf16::relu_bf16,    VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_precision(OpKind::StepElementwise,    &u(bf16_d), vk, unary_bf16::step_bf16,    VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_precision(OpKind::ExpElementwise,     &u(bf16_d), vk, unary_bf16::exp_bf16,     VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_precision(OpKind::LogElementwise,     &u(bf16_d), vk, unary_bf16::log_bf16,     VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_precision(OpKind::SinElementwise,     &u(bf16_d), vk, unary_bf16::sin_bf16,     VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_precision(OpKind::CosElementwise,     &u(bf16_d), vk, unary_bf16::cos_bf16,     VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_precision(OpKind::TanhElementwise,    &u(bf16_d), vk, unary_bf16::tanh_bf16,    VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_precision(OpKind::SigmoidElementwise, &u(bf16_d), vk, unary_bf16::sigmoid_bf16, VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_precision(OpKind::SiluElementwise,    &u(bf16_d), vk, unary_bf16::silu_bf16,    VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_precision(OpKind::GeluElementwise,    &u(bf16_d), vk, unary_bf16::gelu_bf16,    VULKAN_TRANSCENDENTAL_PRECISION);
    table.register_with_precision(OpKind::AbsElementwise,     &u(bf16_d), vk, unary_bf16::abs_bf16,     VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_precision(OpKind::SignElementwise,    &u(bf16_d), vk, unary_bf16::sign_bf16,    VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_precision(OpKind::RecipElementwise,   &u(bf16_d), vk, unary_bf16::recip_bf16,   VULKAN_HALF_POINTWISE_PRECISION);

    // ----- Triu / Tril — pure byte-level mask kernel (rank-3 reshape).
    // CONTIGUOUS-ONLY by design: triu_b4 / tril_b4 view inputs as a
    // flat (outer, dim_size, inner) 3-tuple. Arbitrary layout strides
    // would require a different decomposition; auto-Contiguize handles
    // non-contiguous inputs upstream.
    //
    // Flip / Roll — STRIDE-AWARE (alpha.31 sweep follow-up): the
    // flip_b* / roll_b* kernels now walk rank-N + per-input strides
    // with the axis from OpParams::{Flip, Roll}.axis. Output is contig
    // over the input's shape.
    for &dt in &[DType::F32, DType::F16, DType::BF16, DType::F64,
                 DType::I32, DType::U32, DType::I64] {
        table.register_with_precision(OpKind::Triu, &u(dt), vk, triangular::triu, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::Tril, &u(dt), vk, triangular::tril, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_caps_and_precision(OpKind::Flip, &u(dt), vk, flip::flip, strided, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_caps_and_precision(OpKind::Roll, &u(dt), vk, roll::roll, strided, VULKAN_BYTE_LEVEL_PRECISION);
    }

    // ----- CumSum (inclusive prefix sum along one axis) — per-dtype
    // because the accumulator needs typed addition. Sequential per-
    // slice walk; stride-aware (rank-N + axis from OpParams::CumSum).
    // F32/F64 accumulate in their native types; F16 accumulates in
    // f16 (matches CPU/CUDA semantics — caller casts up to f32 first
    // if they want long-sum stability); BF16 accumulates in f32 with
    // bit-level bf16↔f32 conversion at the edges.
    table.register_with_caps_and_precision(OpKind::CumSum, &u(f32),  vk, cumsum::cumsum_f32,  strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::CumSum, &u(f64_d), vk, cumsum::cumsum_f64,  strided, VULKAN_FLOAT_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::CumSum, &u(f16),  vk, cumsum::cumsum_f16,  strided, VULKAN_HALF_POINTWISE_PRECISION);
    table.register_with_caps_and_precision(OpKind::CumSum, &u(bf16_d), vk, cumsum::cumsum_bf16, strided, VULKAN_HALF_POINTWISE_PRECISION);

    // ----- QMatMul (Q4_0 / Q4_K_M / Q8_0) — F32 × U32-quant → F32.
    // CONTIGUOUS-ONLY: the quantized weight stream has a fixed block
    // layout (per-block scale + N quantized lanes); arbitrary strides
    // on the weight buffer would break the dequant kernel's block
    // walk. Activations could in principle be strided but are
    // contiguous in practice — extending is a follow-up if profiles
    // show repeated auto-Contiguize hits on the activation side. -----
    table.register_with_precision(
        OpKind::QMatMul,
        &[DType::F32, DType::U32, DType::F32],
        vk,
        qmatmul::qmatmul_vk,
        VULKAN_QMATMUL_PRECISION,
    );

    // ----- Conv2D f32 (V.3.I) — im2col + matmul. Numerical character
    // matches f32 matmul (per-output deterministic FMA accumulation).
    // CONTIGUOUS-ONLY: conv2d_im2col reads NCHW with the kernel's
    // assumed canonical stride layout. The im2col view materializes
    // a contiguous buffer for the matmul stage; a strided input
    // would require either an explicit pre-Contiguize or an
    // im2col-with-strides variant (significant kernel work). -----
    table.register_with_precision(
        OpKind::Conv2D,
        &[DType::F32, DType::F32, DType::F32],
        vk,
        conv2d::conv2d_f32,
        VULKAN_MATMUL_PRECISION,
    );
    // ----- FlashAttn f32 (V.3 attention) — naive single-pass kernel
    // (NOT tiled FA-2). Supports GQA, causal mask, softmax_scale,
    // alibi. window_left/window_right/softcap bail at the wrapper
    // for now. Both 3-input (q,k,v) and 4-input (q,k,v,alibi)
    // binding shapes register. -----
    table.register_with_precision(
        OpKind::FlashAttn,
        &[DType::F32, DType::F32, DType::F32, DType::F32],
        vk,
        flash_attn::flash_attn_f32,
        VULKAN_FLOAT_POINTWISE_PRECISION,
    );
    table.register_with_precision(
        OpKind::FlashAttn,
        &[DType::F32, DType::F32, DType::F32, DType::F32, DType::F32],
        vk,
        flash_attn::flash_attn_f32,
        VULKAN_FLOAT_POINTWISE_PRECISION,
    );
    {
        let bf16_d = DType::BF16;
        let f16_d = DType::F16;
        table.register_with_precision(
            OpKind::FlashAttn,
            &[bf16_d, bf16_d, bf16_d, bf16_d],
            vk,
            flash_attn::flash_attn_bf16,
            VULKAN_HALF_POINTWISE_PRECISION,
        );
        table.register_with_precision(
            OpKind::FlashAttn,
            &[bf16_d, bf16_d, bf16_d, bf16_d, bf16_d],
            vk,
            flash_attn::flash_attn_bf16,
            VULKAN_HALF_POINTWISE_PRECISION,
        );
        table.register_with_precision(
            OpKind::FlashAttn,
            &[f16_d, f16_d, f16_d, f16_d],
            vk,
            flash_attn::flash_attn_f16,
            VULKAN_HALF_POINTWISE_PRECISION,
        );
        table.register_with_precision(
            OpKind::FlashAttn,
            &[f16_d, f16_d, f16_d, f16_d, f16_d],
            vk,
            flash_attn::flash_attn_f16,
            VULKAN_HALF_POINTWISE_PRECISION,
        );
    }

    // ----- FlashAttn backward Q/K/V, f32. Vulkan kernels for the
    // three OpKinds added at 3fd33aae. Same window/softcap caveats as
    // forward; bf16/f16 backward kernels are own session. Binding key
    // shape: 4-input (q,k,v,do,out) + 5-input (q,k,v,do,alibi,out). -----
    table.register_with_precision(
        OpKind::FlashAttnBackwardQ,
        &[DType::F32, DType::F32, DType::F32, DType::F32, DType::F32],
        vk,
        flash_attn::flash_attn_backward_q_f32,
        VULKAN_FLOAT_POINTWISE_PRECISION,
    );
    table.register_with_precision(
        OpKind::FlashAttnBackwardQ,
        &[DType::F32, DType::F32, DType::F32, DType::F32, DType::F32, DType::F32],
        vk,
        flash_attn::flash_attn_backward_q_f32,
        VULKAN_FLOAT_POINTWISE_PRECISION,
    );
    table.register_with_precision(
        OpKind::FlashAttnBackwardK,
        &[DType::F32, DType::F32, DType::F32, DType::F32, DType::F32],
        vk,
        flash_attn::flash_attn_backward_k_f32,
        VULKAN_FLOAT_POINTWISE_PRECISION,
    );
    table.register_with_precision(
        OpKind::FlashAttnBackwardK,
        &[DType::F32, DType::F32, DType::F32, DType::F32, DType::F32, DType::F32],
        vk,
        flash_attn::flash_attn_backward_k_f32,
        VULKAN_FLOAT_POINTWISE_PRECISION,
    );
    table.register_with_precision(
        OpKind::FlashAttnBackwardV,
        &[DType::F32, DType::F32, DType::F32, DType::F32, DType::F32],
        vk,
        flash_attn::flash_attn_backward_v_f32,
        VULKAN_FLOAT_POINTWISE_PRECISION,
    );
    table.register_with_precision(
        OpKind::FlashAttnBackwardV,
        &[DType::F32, DType::F32, DType::F32, DType::F32, DType::F32, DType::F32],
        vk,
        flash_attn::flash_attn_backward_v_f32,
        VULKAN_FLOAT_POINTWISE_PRECISION,
    );

    // ----- Conv2D bf16 (V.3.I extended) — im2col_bf16 + cooperative-
    // matrix bf16 matmul (matmul_coop_bf16_bf16_bf16). Activations,
    // weights, and output all bf16. COOP-ONLY shape constraints:
    // c_out % 16 == 0 && (h_out * w_out) % 16 == 0. Wider ULP than
    // f32 conv2d due to bf16→f16 downcast on matmul inputs; the
    // tensor-core precision tag reflects that. -----
    table.register_with_precision(
        OpKind::Conv2D,
        &[DType::BF16, DType::BF16, DType::BF16],
        vk,
        conv2d::conv2d_bf16,
        VULKAN_MATMUL_TENSORCORE_PRECISION,
    );
    // ----- Conv2D f16 — sibling of bf16. Reuses the bf16 im2col
    // shader (2-byte dtype-opaque shuffle) + matmul_coop_f16_f16_f16.
    // Same shape constraints. -----
    table.register_with_precision(
        OpKind::Conv2D,
        &[DType::F16, DType::F16, DType::F16],
        vk,
        conv2d::conv2d_f16,
        VULKAN_MATMUL_TENSORCORE_PRECISION,
    );

    // ----- Cast F8E4M3 ↔ {F32, F16, BF16} — pure dtype conversion.
    // strided-input candidate (with caveat): F8E4M3 packs 4 elements
    // per u32 (1 byte each), so element count must be a multiple of 4
    // and stride arithmetic gets awkward at the packing boundary.
    // A strided extension is plausible but requires careful handling
    // of the 4-element-aligned pack/unpack. -----
    let f8 = DType::F8E4M3;
    table.register_with_precision(OpKind::Cast, &[f32,    f8],     vk, cast_f8e4m3::cast_f8e4m3, VULKAN_CAST_PRECISION);
    table.register_with_precision(OpKind::Cast, &[f8,     f32],    vk, cast_f8e4m3::cast_f8e4m3, VULKAN_CAST_PRECISION);
    table.register_with_precision(OpKind::Cast, &[f16,    f8],     vk, cast_f8e4m3::cast_f8e4m3, VULKAN_CAST_PRECISION);
    table.register_with_precision(OpKind::Cast, &[f8,     f16],    vk, cast_f8e4m3::cast_f8e4m3, VULKAN_CAST_PRECISION);
    table.register_with_precision(OpKind::Cast, &[bf16_d, f8],     vk, cast_f8e4m3::cast_f8e4m3, VULKAN_CAST_PRECISION);
    table.register_with_precision(OpKind::Cast, &[f8,     bf16_d], vk, cast_f8e4m3::cast_f8e4m3, VULKAN_CAST_PRECISION);

    // ----- Op::Copy D2H (bridge-retirement Phase 2). Byte-level.
    // CONTIGUOUS-ONLY: the wrapper downloads the source's bytes via
    // a Vulkan staging buffer (vkCmdCopyBuffer). The staging buffer
    // is sized to the source's flat byte count; stride-aware D2H
    // would require either per-row staging or an explicit pre-
    // Contiguize step. Auto-Contiguize handles non-contiguous
    // sources upstream of this kernel. -----
    let copy_dtypes = [
        f32, f16, bf16_d, DType::F64, DType::U32, DType::U8,
        DType::I16, DType::I32, DType::I64,
    ];
    for dt in copy_dtypes {
        table.register_with_precision(OpKind::Copy, &[dt, dt], vk, copy_to_cpu_vulkan, VULKAN_BYTE_LEVEL_PRECISION);
    }

    // ----- Bulk-fill cost functions for every Vulkan registration above.
    // The CPU dispatcher (`default_cost_for_op_kind`) captures the
    // FLOP/bandwidth model; backend-specific kernel_overhead_ns
    // adjustments are deferred to a Vulkan-flavored dispatcher (or to
    // the empirical calibration framework). Every entry above started
    // at `unknown_cost`; this pass upgrades them to a real cost
    // function so the optimizer's cost-ranking can admit Vulkan
    // alternatives.
    table.fill_unset_cost_for_backend(vk, crate::cost::default_cost_for_op_kind);
}
