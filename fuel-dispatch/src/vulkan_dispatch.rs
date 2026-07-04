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

use fuel_ir::dispatch::OpKind;
use fuel_ir::probe::BackendId;
use fuel_ir::{DType, Error, Layout, Result};

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
        ) -> fuel_ir::Result<()>,
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
        ) -> fuel_ir::Result<()>,
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
        ) -> fuel_ir::Result<()>,
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
        ) -> fuel_ir::Result<()>,
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
                    OpParams::WriteSlice { dest_shape, ranges, .. } => (dest_shape, ranges),
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
            OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k, .. } => {
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
            OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k, .. } => {
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
            OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k, .. } => {
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
            OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k, .. } => {
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
            OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k, .. } => {
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
            OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k, .. } => {
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
    use fuel_ir::Shape;

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
        ) -> fuel_ir::Result<()>,
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
        ) -> fuel_ir::Result<()>,
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

/// The authored Vulkan cast (dtype-conversion) kernel contract, embedded into
/// the binary via `include_str!` (the PRODUCTION contract). Parsed + lowered by
/// [`register_vulkan_cast_from_contract`].
const VULKAN_CAST_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/vulkan/cast.fkc.md");

/// Register the Vulkan cast family (12 (SRC, DST) pairs → 3 production wrappers:
/// `cast::cast_f32_half` for f32↔f16 / f32↔bf16, `cast::cast_f32_f64` for
/// f32↔f64, `cast_f8e4m3::cast_f8e4m3` for the six F8E4M3 pairs) by IMPORTING
/// its FKC kernel contract — the **first Vulkan-backend FKC consumer**, the GPU
/// analogue of the CPU `register_cpu_*_from_contract` family. FKC is
/// unconditional core infrastructure, so this is the ONE registration path for
/// the family: the hand-written `table.register_with_precision(OpKind::Cast, …)`
/// calls (both the half/f64 block and the F8E4M3 block) are DELETED.
///
/// Each per-pair section declares a single-dtype `src` input + a `fixed(DST)`
/// output, so the importer keys `[SRC, DST]` — byte-for-byte the deleted regs
/// (the `cast(output)` dtype-rule the sections used to carry was NOT an
/// importer-recognized form and would have dropped the output slot from the
/// key; it was rewritten to the canonical `fixed(DST)` per section's documented
/// destination). Every `entry_point` symbol resolves through the production
/// [`crate::fkc::VulkanLinkRegistry`] to the exact same wrapper fn-pointer.
///
/// Behavior-preserving vs. the deleted hand-written path on dispatch: identical
/// wrappers + contiguous-only caps (`awkward_layout_strategy: requires_contiguous`
/// ⇒ `strided_input == false`). The binding's `kernel_source` becomes the
/// contract's `"vulkan-slang"` tag and its precision the contract's audited
/// claim — the shared `VULKAN_CAST_PRECISION` const the regs used to carry
/// (`max_ulp: Some(0)`) is retired; the contract's per-section `max_ulp: ~`
/// author seed is what the Judge now audits (both keep
/// `bit_stable_on_same_hardware: true`). Cost is preserved because this runs
/// BEFORE the `fill_unset_cost_for_backend` pass, which upgrades the imported
/// entries' `unknown_cost` sentinel to the same OpKind cost fn every Vulkan
/// primitive gets.
///
/// The family declares NO fused ops, so `register_into`'s required fused
/// argument is a local throwaway that provably stays empty. Init-boundary
/// fail-fast: a parse/lower/link failure of the embedded, authored contract is
/// a programmer error surfaced once here via `expect` (mirroring the CPU
/// `register_cpu_*_from_contract` convention); it cannot fail for a runtime-data
/// reason.
fn register_vulkan_cast_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(VULKAN_CAST_CONTRACT, &crate::fkc::VulkanLinkRegistry)
            .expect(
                "authored Vulkan cast contract must import \
                 (embedded via include_str!, resolved through VulkanLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "vulkan cast contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider
        .register_into(table, &mut fused)
        .expect("Vulkan cast contract must register into the binding table");
}

/// The authored Vulkan elementwise kernel contract (unary / binary / affine /
/// clamp / powi), embedded via `include_str!` (the PRODUCTION contract). Parsed
/// + lowered by [`register_vulkan_elementwise_from_contract`].
const VULKAN_ELEMENTWISE_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/vulkan/elementwise.fkc.md");

/// Register the Vulkan elementwise family (94 (op, dtype) bindings) by IMPORTING
/// its RE-AUTHORED FKC kernel contract — the SECOND Vulkan-backend FKC consumer
/// after the cast family, and the first **caps-through-import** production proof.
///
/// The contract was re-authored per-op (the CPU inplace precedent): the old
/// representative-chassis `unary`/`binary` op_id-selector sections could each
/// register only ONE OpKind, so they are SUPERSEDED by 16 per-op unary sections
/// (each fanning `[F32, F16, F64]` on a base `entry_point`, §3.4) + 16 bf16-unary
/// sections + 6 per-op binary sections (fanning `[F32, F16, F64, BF16]`) + an
/// affine strided-fan + affine_bf16 + clamp + powi. Every `entry_point` symbol
/// resolves through [`crate::fkc::VulkanLinkRegistry`] to the exact same
/// per-(op, dtype) wrapper the deleted hand-written regs used.
///
/// **Caps ride through truthfully.** Each section's per-operand five-flag layout
/// projects onto `KernelCaps.strided_input` (`caps_map::project_kernel_caps`),
/// stamped onto the binding by the importer: the strided unary/binary/affine/
/// clamp/powi sections yield `strided_input=true` (byte-for-byte the deleted
/// `register_with_caps_and_precision(strided)` regs) and the contiguous-only
/// bf16-unary / affine_bf16 sections yield `strided_input=false` (the deleted
/// plain `register_with_precision` regs). Precision becomes the contract's author
/// seed (`audited: false` ⇒ `PrecisionGuarantee::UNAUDITED`; the Judge audits
/// later) — the same posture the cast migration took; the hand-written
/// `VULKAN_{FLOAT,HALF,TRANSCENDENTAL}_POINTWISE_PRECISION` consts are retired
/// from this seam. Cost is preserved because this runs BEFORE
/// `fill_unset_cost_for_backend`, which upgrades the imported `unknown_cost`
/// sentinels to the same OpKind cost fn.
///
/// The contract's `add_assign_scaled` + `unary`/`binary` chassis sections are
/// `registrable: false` describe-only (§3.10) and register nothing. Init-boundary
/// fail-fast: a parse/lower/link failure of the embedded contract is a programmer
/// error surfaced once here via `expect`.
fn register_vulkan_elementwise_from_contract(table: &mut KernelBindingTable) {
    let provider = crate::fkc::import_bundle_str(
        VULKAN_ELEMENTWISE_CONTRACT,
        &crate::fkc::VulkanLinkRegistry,
    )
    .expect(
        "authored Vulkan elementwise contract must import \
         (embedded via include_str!, resolved through VulkanLinkRegistry)",
    );
    debug_assert!(
        provider.fused.is_empty(),
        "vulkan elementwise contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider
        .register_into(table, &mut fused)
        .expect("Vulkan elementwise contract must register into the binding table");
}

/// The authored Vulkan matmul kernel contract (6 per-combo GEMM/GEMV wrapper
/// bindings), embedded via `include_str!` (the PRODUCTION contract). Parsed +
/// lowered by [`register_vulkan_matmul_from_contract`].
const VULKAN_MATMUL_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/vulkan/matmul.fkc.md");

/// Register the Vulkan matmul family (6 (MatMul, [lhs,rhs,out]) bindings) by
/// IMPORTING its RE-AUTHORED FKC kernel contract — the THIRD Vulkan-backend FKC
/// consumer after cast + elementwise. FKC is unconditional core infrastructure,
/// so this is the ONE registration path for the family: the hand-written
/// `table.register_with_precision(OpKind::MatMul, …)` regs (the f32 GEMM +
/// the five mixed-precision / tensor-core combos) are DELETED.
///
/// The contract was re-authored per-combo (the cast family's per-pair
/// precedent): the aspirational multi-dtype `dispatch/matmul.fkc.md ::
/// matmul_mixed_precision` section (lhs `[F32,BF16,F16]` vs rhs `[BF16,F16]` —
/// DIFFERENT lists, a `FanoutDtypeMismatch` the uniform fan-out importer cannot
/// key) is SUPERSEDED by six single-dtype-per-operand sections, each declaring a
/// specific `entry_point` symbol that resolves through
/// [`crate::fkc::VulkanLinkRegistry`] to the exact production wrapper. Each
/// wrapper route-picks its internal Slang kernels (matvec / reg-tile / tiled for
/// f32; matvec_bf16_b / matmul_tiled_bf16_b / matmul_coop* + matmul_small_*
/// fallbacks for the mixed combos) at dispatch time — that route-pick is NOT a
/// binding, so there is exactly ONE `KernelRef` per key.
///
/// **Caps ride through truthfully.** Each section's `requires_contiguous`
/// layout projects `strided_input == false` — byte-for-byte the deleted
/// `register_with_precision` regs (the coop / vec4 loads require canonical
/// row-major; a strided / transposed / offset operand is auto-Contiguized by the
/// planner first). Precision becomes the contract's audited seed
/// (nondeterministic, `bit_stable_on_same_hardware: false`, audited `none(reason)`
/// — the corrected 2026-06-18 posture, honest about scheduler-dependent
/// FADD/subgroup order) rather than the retired hand-written
/// `VULKAN_MATMUL_PRECISION` / `VULKAN_MATMUL_TENSORCORE_PRECISION` consts (which
/// mis-declared `bit_stable_on_same_hardware: true`). At the matmul migration
/// those consts were left in use by the Vulkan conv2d family; conv2d has since
/// migrated to its own FKC contract, so they are now dropped from this seam too
/// (the `pub const` defs stay in `fused.rs`, mirroring VULKAN_TRANSCENDENTAL).
/// Cost is preserved because this runs BEFORE `fill_unset_cost_for_backend`,
/// which upgrades the imported `unknown_cost` sentinels to the same OpKind cost fn.
///
/// The family declares NO fused ops. Init-boundary fail-fast: a parse/lower/link
/// failure of the embedded, authored contract is a programmer error surfaced once
/// here via `expect`.
fn register_vulkan_matmul_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(VULKAN_MATMUL_CONTRACT, &crate::fkc::VulkanLinkRegistry)
            .expect(
                "authored Vulkan matmul contract must import \
                 (embedded via include_str!, resolved through VulkanLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "vulkan matmul contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider
        .register_into(table, &mut fused)
        .expect("Vulkan matmul contract must register into the binding table");
}

/// The authored Vulkan conv2d kernel contract (3 per-(op, dtype) Conv2D
/// bindings), embedded via `include_str!` (the PRODUCTION contract). Parsed +
/// lowered by [`register_vulkan_conv_from_contract`].
const VULKAN_CONV_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/vulkan/conv.fkc.md");

/// Register the Vulkan conv2d family (3 `(Conv2D, [x, weight, out])` bindings —
/// f32 / bf16 / f16) by IMPORTING its RE-AUTHORED FKC kernel contract — the
/// FOURTH Vulkan-backend FKC consumer after cast + elementwise + matmul. FKC is
/// unconditional core infrastructure, so this is the ONE registration path for
/// the family: the hand-written `table.register_with_precision(OpKind::Conv2D,
/// …)` regs (f32 + the bf16 / f16 cooperative-matrix combos) are DELETED.
///
/// The contract was RE-AUTHORED per-(op, dtype) (the matmul family's per-combo
/// precedent): the corpus's `vulkan/conv-attn-rope.fkc.md` describes conv2d as an
/// ASPIRATIONAL `fused_op: CONV2D` **im2col STAGE** (`conv2d_im2col_f32` /
/// `conv2d_im2col_bf16`, whose output is the intermediate patches matrix, NOT the
/// conv output) — that future *fused* decomposition is a SEPARATE concern and does
/// NOT register the primitive `OpKind::Conv2D` binding production actually wires,
/// whose wrapper (`conv2d::conv2d_f32` → `VulkanBackend::conv2d_*_bytes`) runs the
/// WHOLE conv (im2col + GEMM internally). So the migration authors THREE faithful
/// single-dtype-per-operand `op_kind: Conv2D` sections in the new production
/// `docs/kernel-contracts/vulkan/conv.fkc.md`, each `entry_point` resolving through
/// [`crate::fkc::VulkanLinkRegistry`] to the exact production wrapper.
///
/// **No bias key.** The Vulkan conv wrappers BAIL on a 3-input (bias) call, so —
/// UNLIKE the CPU conv contract's `optional: true` bias that fans a `[T,T,T,T]`
/// with-bias key — the sections declare NO bias operand and each keys ONLY the
/// 3-slot `[x, weight, out]`, byte-for-byte the deleted hand-written regs.
///
/// **Caps ride through truthfully.** Each section's `requires_contiguous` layout
/// projects `strided_input == false` — byte-for-byte the deleted
/// `register_with_precision` regs (conv2d_im2col reads canonical row-major NCHW;
/// a strided / transposed / offset operand is auto-Contiguized by the planner
/// first). Precision becomes the contract's audited seed (nondeterministic,
/// `bit_stable_on_same_hardware: false`, audited `none(reason)` — the corrected,
/// honest posture) rather than the retired hand-written `VULKAN_MATMUL_PRECISION`
/// / `VULKAN_MATMUL_TENSORCORE_PRECISION` consts (which mis-declared
/// `bit_stable_on_same_hardware: true`; conv was their LAST code user, so they are
/// dropped from this seam — the matmul migration deferred that retirement to here).
/// Cost is preserved because this runs BEFORE `fill_unset_cost_for_backend`, which
/// upgrades the imported `unknown_cost` sentinels to the same OpKind cost fn.
///
/// The family declares NO fused ops. Init-boundary fail-fast: a parse/lower/link
/// failure of the embedded, authored contract is a programmer error surfaced once
/// here via `expect`.
fn register_vulkan_conv_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(VULKAN_CONV_CONTRACT, &crate::fkc::VulkanLinkRegistry)
            .expect(
                "authored Vulkan conv contract must import \
                 (embedded via include_str!, resolved through VulkanLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "vulkan conv contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider
        .register_into(table, &mut fused)
        .expect("Vulkan conv contract must register into the binding table");
}

/// The authored Vulkan QMatMul kernel contract (ONE `(QMatMul, [F32, U32, F32])`
/// binding), embedded via `include_str!` (the PRODUCTION contract). Parsed +
/// lowered by [`register_vulkan_qmatmul_from_contract`].
const VULKAN_QMATMUL_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/vulkan/qmatmul.fkc.md");

/// Register the Vulkan QMatMul binding (ONE `(QMatMul, [F32, U32, F32])` key) by
/// IMPORTING its RE-AUTHORED FKC kernel contract — the FOURTEENTH Vulkan-backend
/// FKC consumer. FKC is unconditional core infrastructure, so this is the ONE
/// registration path for the family: the hand-written
/// `table.register_with_precision(OpKind::QMatMul, &[F32, U32, F32], …)` reg is
/// DELETED.
///
/// The corpus `vulkan/quantized.fkc.md` models QMATMUL as TWO aspirational
/// `fused_op: QMATMUL` Q4_0 sections (`qmatvec_q4_0` gemv + `matmul_q4_0_tiled`
/// prefill), which mismatch production on BOTH axes: production registers the
/// PRIMITIVE `OpKind::QMatMul` (not the fused op) at a key whose weight slot is the
/// LOGICAL `U32` dispatch dtype (not the physical `U8` block stream). So the
/// migration authors ONE faithful single-dtype-per-operand `op_kind: QMatMul`
/// section in the new production `docs/kernel-contracts/vulkan/qmatmul.fkc.md`
/// (the CPU quant-matmul primitive precedent), applying the maintainer-approved
/// CPU linear-quant reconciliation:
///  - **weight U8 → U32**: `accept.dtypes` carries the LOGICAL dispatch dtype the
///    binding key + `BackendImpl.dtypes` actually use (`[F32, U32, F32]`, per the
///    DType-logical / SType-physical split in `docs/specs/storage-encoding.md`);
///    the `fdx.quant` GGML_BLOCK block keeps the physical byte-honesty
///    (`quant_coherence` does not pin the operand base dtype for GGML_BLOCK).
/// and flips the two corpus fused sections to `registrable: false` describe-only
/// cross-referencing it (the matmul_mixed_precision precedent).
///
/// The section's `entry_point` resolves through [`crate::fkc::VulkanLinkRegistry`]
/// to the exact production wrapper `qmatmul::qmatmul_vk`, which route-picks the
/// per-format kernels (`qmatvec_q4_0` / `matmul_q4_0_tiled` for Q4_0;
/// dequant-then-`matmul_f32` for Q4_K_M / Q8_0) at dispatch time by
/// `OpParams::QMatMul.quant_type` — that route-pick is NOT a binding, so there is
/// exactly ONE `KernelRef` at the key.
///
/// **Caps ride through truthfully.** The section's `requires_contiguous` layout
/// projects `strided_input == false` — byte-for-byte the deleted
/// `register_with_precision` reg (the GGML weight stream has a fixed per-block
/// layout; a strided / transposed / offset operand is auto-Contiguized by the
/// planner first). Precision becomes the contract's audited seed (nondeterministic,
/// `bit_stable_on_same_hardware: false`, audited `none(reason)` — the corrected,
/// honest posture: the Q4_0 kernels reduce over K with a scheduler-dependent
/// subgroup / tiled reduction, and the Q4_K_M / Q8_0 routes contract through the
/// same `matmul_f32` GEMM) rather than the retired hand-written
/// `VULKAN_QMATMUL_PRECISION` const (which mis-declared
/// `bit_stable_on_same_hardware: true`, the same over-claim the retired
/// `VULKAN_MATMUL_PRECISION` made; the `pub const` def stays in `fused.rs`,
/// mirroring VULKAN_TRANSCENDENTAL). Cost is preserved because this runs BEFORE
/// `fill_unset_cost_for_backend`, which upgrades the imported `unknown_cost`
/// sentinel to the same OpKind cost fn.
///
/// The family declares NO fused ops. Init-boundary fail-fast: a parse/lower/link
/// failure of the embedded, authored contract is a programmer error surfaced once
/// here via `expect`.
fn register_vulkan_qmatmul_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(VULKAN_QMATMUL_CONTRACT, &crate::fkc::VulkanLinkRegistry)
            .expect(
                "authored Vulkan qmatmul contract must import \
                 (embedded via include_str!, resolved through VulkanLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "vulkan qmatmul contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider
        .register_into(table, &mut fused)
        .expect("Vulkan qmatmul contract must register into the binding table");
}

/// The authored Vulkan select (IndexSelect / Gather / MaskedFill) kernel
/// contract, embedded via `include_str!` (the PRODUCTION contract). Parsed +
/// lowered by [`register_vulkan_select_from_contract`].
const VULKAN_SELECT_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/vulkan/select.fkc.md");

/// Register the Vulkan select family (16 (op, dtype) bindings — IndexSelect (4),
/// Gather (6), MaskedFill (6)) by IMPORTING its FKC kernel contract — the FIFTH
/// Vulkan-backend FKC consumer after cast + elementwise + matmul + conv. FKC is
/// unconditional core infrastructure, so this is the ONE registration path for
/// the family: the hand-written `register_with_precision(OpKind::{IndexSelect,
/// Gather, MaskedFill}, …)` regs are DELETED.
///
/// Each section fans a BASE `entry_point` over its dtype list (§3.4): IndexSelect
/// resolves `index_select_<suffix>` to FOUR distinct per-dtype wrappers; Gather /
/// MaskedFill resolve every `<base>_<suffix>` to the ONE dtype-agnostic wrapper
/// (`gather::gather` / `masked_fill::masked_fill`) that picks its element
/// byte-width from the output dtype — a synthetic-base umbrella (the pad_cpu
/// precedent). The importer keys `[data, index, out]` (data dtype + fixed index
/// dtype U32/U8 + `passthrough(data)` output), byte-for-byte the deleted regs.
///
/// **Caps ride through truthfully.** Each section's `requires_contiguous` layout
/// projects `strided_input == false` — byte-for-byte the deleted plain
/// `register_with_precision` regs (byte-level data movers read/write own-shape
/// flat buffers; a strided operand is auto-Contiguized first). Precision is the
/// contract's audited byte-exact seed (`bit_stable`, `max_ulp: 0`,
/// `determinism: bitwise`) — the honest posture for a pure byte copy, byte-for-byte
/// the deleted regs' `VULKAN_BYTE_LEVEL_PRECISION`. Cost is preserved because this
/// runs BEFORE `fill_unset_cost_for_backend`, which upgrades the imported
/// `unknown_cost` sentinels to the same OpKind cost fn.
///
/// The family declares NO fused ops. Init-boundary fail-fast: a parse/lower/link
/// failure of the embedded, authored contract is a programmer error surfaced once
/// here via `expect`.
fn register_vulkan_select_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(VULKAN_SELECT_CONTRACT, &crate::fkc::VulkanLinkRegistry)
            .expect(
                "authored Vulkan select contract must import \
                 (embedded via include_str!, resolved through VulkanLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "vulkan select contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider
        .register_into(table, &mut fused)
        .expect("Vulkan select contract must register into the binding table");
}

/// The authored Vulkan scatter (IndexAdd / ScatterAdd) kernel contract, embedded
/// via `include_str!` (the PRODUCTION contract). Parsed + lowered by
/// [`register_vulkan_scatter_from_contract`].
const VULKAN_SCATTER_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/vulkan/scatter.fkc.md");

/// Register the Vulkan scatter family (8 (op, dtype) bindings — IndexAdd (4) +
/// ScatterAdd (4)) by IMPORTING its FKC kernel contract — the SIXTH Vulkan-backend
/// FKC consumer. FKC is unconditional core infrastructure, so this is the ONE
/// registration path for the family: the hand-written
/// `register_with_precision(OpKind::{IndexAdd,ScatterAdd}, …)` regs are DELETED.
///
/// Each section fans a BASE `entry_point` over `[F32, F64, BF16, F16]` (the `base`
/// + `src` operands share the list ⇒ they fan together, §3.4), resolving
/// `<base>_<suffix>` to each distinct per-dtype wrapper and keying the 4-slot
/// `[base, U32, src, out]` (`passthrough(base)` output), byte-for-byte the deleted
/// regs. Caps ride through contiguous-only (`requires_contiguous` ⇒
/// `strided_input == false`). Precision is the contract's audited nondeterministic
/// `none(reason)` seed (bounded-CAS atomic accumulate, scheduler-dependent order)
/// — byte-for-byte the deleted regs' `PrecisionGuarantee::none(reason)`. Cost is
/// preserved because this runs BEFORE `fill_unset_cost_for_backend`.
///
/// The family declares NO fused ops. Init-boundary fail-fast: a parse/lower/link
/// failure of the embedded, authored contract is a programmer error surfaced once
/// here via `expect`.
fn register_vulkan_scatter_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(VULKAN_SCATTER_CONTRACT, &crate::fkc::VulkanLinkRegistry)
            .expect(
                "authored Vulkan scatter contract must import \
                 (embedded via include_str!, resolved through VulkanLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "vulkan scatter contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider
        .register_into(table, &mut fused)
        .expect("Vulkan scatter contract must register into the binding table");
}

/// The authored Vulkan movement (Concat / CumSum) kernel contract, embedded via
/// `include_str!` (the PRODUCTION contract). Parsed + lowered by
/// [`register_vulkan_movement_from_contract`].
const VULKAN_MOVEMENT_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/vulkan/movement.fkc.md");

/// Register the Vulkan movement family (8 (op, dtype) bindings — Concat (4) +
/// CumSum (4)) by IMPORTING its FKC kernel contract — the SEVENTH Vulkan-backend
/// FKC consumer. FKC is unconditional core infrastructure, so this is the ONE
/// registration path for the family: the hand-written
/// `register_with_caps_and_precision(OpKind::{Concat,CumSum}, …, strided, …)` regs
/// are DELETED.
///
/// Each section fans a BASE `entry_point` over its dtype list (§3.4), resolving
/// `<base>_<suffix>` to each distinct per-dtype wrapper and keying `[T, T]`
/// (`passthrough(input)` output), byte-for-byte the deleted regs. Both are
/// STRIDE-AWARE (`strided: accepted` ⇒ `strided_input == true`) — the
/// caps-through-import proof for this family. Concat precision is byte-exact
/// (bitwise); CumSum takes the conservative UNAUDITED author seed (the elementwise
/// pointwise precedent — the retired `VULKAN_{FLOAT,HALF}_POINTWISE_PRECISION`
/// consts are not re-asserted). Cost is preserved because this runs BEFORE
/// `fill_unset_cost_for_backend`.
///
/// The family declares NO fused ops. Init-boundary fail-fast: a parse/lower/link
/// failure of the embedded, authored contract is a programmer error surfaced once
/// here via `expect`.
fn register_vulkan_movement_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(VULKAN_MOVEMENT_CONTRACT, &crate::fkc::VulkanLinkRegistry)
            .expect(
                "authored Vulkan movement contract must import \
                 (embedded via include_str!, resolved through VulkanLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "vulkan movement contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider
        .register_into(table, &mut fused)
        .expect("Vulkan movement contract must register into the binding table");
}

/// The authored Vulkan shape (Triu / Tril / Flip / Roll) kernel contract, embedded
/// via `include_str!` (the PRODUCTION contract). Parsed + lowered by
/// [`register_vulkan_shape_from_contract`].
const VULKAN_SHAPE_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/vulkan/shape.fkc.md");

/// Register the Vulkan shape family (28 (op, dtype) bindings — Triu (7) + Tril (7)
/// + Flip (7) + Roll (7)) by IMPORTING its FKC kernel contract — the EIGHTH
/// Vulkan-backend FKC consumer. FKC is unconditional core infrastructure, so this
/// is the ONE registration path for the family: the hand-written
/// `register_with_precision(OpKind::{Triu,Tril}, …)` +
/// `register_with_caps_and_precision(OpKind::{Flip,Roll}, …, strided, …)` regs are
/// DELETED.
///
/// Each op is ONE dtype-agnostic wrapper across its 7 dtype keys; each section fans
/// the BASE `entry_point` over `[F32, F16, BF16, F64, I32, U32, I64]` and the link
/// registry maps every `<base>_<suffix>` symbol to the one wrapper (a
/// synthetic-base umbrella, the pad_cpu precedent), keying `[T, T]` byte-for-byte
/// the deleted regs. Caps split by op: Triu/Tril contiguous
/// (`strided_input == false`), Flip/Roll stride-aware (`strided_input == true`) —
/// the caps-through-import proof. Precision is byte-exact (bitwise, `max_ulp: 0`),
/// byte-for-byte the deleted regs' `VULKAN_BYTE_LEVEL_PRECISION`. Cost is preserved
/// because this runs BEFORE `fill_unset_cost_for_backend`.
///
/// The family declares NO fused ops. Init-boundary fail-fast: a parse/lower/link
/// failure of the embedded, authored contract is a programmer error surfaced once
/// here via `expect`.
fn register_vulkan_shape_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(VULKAN_SHAPE_CONTRACT, &crate::fkc::VulkanLinkRegistry)
            .expect(
                "authored Vulkan shape contract must import \
                 (embedded via include_str!, resolved through VulkanLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "vulkan shape contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider
        .register_into(table, &mut fused)
        .expect("Vulkan shape contract must register into the binding table");
}

/// The authored Vulkan pad-copy (Pad / PadBackward / Copy) kernel contract,
/// embedded via `include_str!` (the PRODUCTION contract). Parsed + lowered by
/// [`register_vulkan_pad_copy_from_contract`].
const VULKAN_PAD_COPY_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/vulkan/pad-copy.fkc.md");

/// Register the Vulkan pad-copy family (21 (op, dtype) bindings — Pad (6) +
/// PadBackward (6) + Copy (9)) by IMPORTING its FKC kernel contract — the NINTH
/// Vulkan-backend FKC consumer. FKC is unconditional core infrastructure, so this
/// is the ONE registration path for the family: the hand-written
/// `register_with_precision(OpKind::{Pad,PadBackward,Copy}, …)` regs are DELETED.
///
/// Each op is ONE dtype-agnostic wrapper across its dtype keys; each section fans
/// the BASE `entry_point` over its dtype list and the link registry maps every
/// `<base>_<suffix>` symbol to the one wrapper (a synthetic-base umbrella, the
/// pad_cpu precedent), keying `[T, T]` byte-for-byte the deleted regs. Caps ride
/// through contiguous-only (`requires_contiguous` ⇒ `strided_input == false`);
/// precision is byte-exact (bitwise, `max_ulp: 0`), byte-for-byte the deleted regs'
/// `VULKAN_BYTE_LEVEL_PRECISION`. Cost is preserved because this runs BEFORE
/// `fill_unset_cost_for_backend`.
///
/// The family declares NO fused ops. Init-boundary fail-fast: a parse/lower/link
/// failure of the embedded, authored contract is a programmer error surfaced once
/// here via `expect`.
fn register_vulkan_pad_copy_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(VULKAN_PAD_COPY_CONTRACT, &crate::fkc::VulkanLinkRegistry)
            .expect(
                "authored Vulkan pad-copy contract must import \
                 (embedded via include_str!, resolved through VulkanLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "vulkan pad-copy contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider
        .register_into(table, &mut fused)
        .expect("Vulkan pad-copy contract must register into the binding table");
}

/// The authored Vulkan write-slice (WriteSlice / WriteSliceRotating) kernel
/// contract, embedded via `include_str!` (the PRODUCTION contract). Parsed +
/// lowered by [`register_vulkan_write_slice_from_contract`].
const VULKAN_WRITE_SLICE_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/vulkan/write-slice.fkc.md");

/// Register the Vulkan write-slice family (18 (op, dtype) bindings — WriteSlice (9)
/// + WriteSliceRotating (9)) by IMPORTING its FKC kernel contract — the TENTH
/// Vulkan-backend FKC consumer. FKC is unconditional core infrastructure, so this
/// is the ONE registration path for the family: the hand-written
/// `register_with_precision(OpKind::{WriteSlice,WriteSliceRotating}, …)` regs are
/// DELETED.
///
/// BYTE-WIDTH-keyed: each op's 9 dtype keys collapse to 4 byte-width wrappers
/// (`b1/b2/b4/b8`) — the cast family's "several sections share one wrapper"
/// precedent. Each section fans the BASE `entry_point` over the 9-dtype list and
/// the link registry maps each `<base>_<suffix>` symbol to the byte-width wrapper
/// for that dtype's size, keying `[T, T]` byte-for-byte the deleted regs. Caps ride
/// through contiguous-only (`requires_contiguous` ⇒ `strided_input == false`);
/// precision is byte-exact (bitwise, `max_ulp: 0`), byte-for-byte the deleted regs'
/// `VULKAN_BYTE_LEVEL_PRECISION`. Cost is preserved because this runs BEFORE
/// `fill_unset_cost_for_backend`.
///
/// The family declares NO fused ops. Init-boundary fail-fast: a parse/lower/link
/// failure of the embedded, authored contract is a programmer error surfaced once
/// here via `expect`.
fn register_vulkan_write_slice_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(VULKAN_WRITE_SLICE_CONTRACT, &crate::fkc::VulkanLinkRegistry)
            .expect(
                "authored Vulkan write-slice contract must import \
                 (embedded via include_str!, resolved through VulkanLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "vulkan write-slice contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider
        .register_into(table, &mut fused)
        .expect("Vulkan write-slice contract must register into the binding table");
}

/// The authored Vulkan rope kernel contract, embedded via `include_str!` (the
/// PRODUCTION contract). Parsed + lowered by [`register_vulkan_rope_from_contract`].
const VULKAN_ROPE_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/vulkan/rope.fkc.md");

/// Register the Vulkan rope family (4 (Rope, [x, cos, sin, out]) per-dtype
/// bindings) by IMPORTING its FKC kernel contract — the ELEVENTH Vulkan-backend
/// FKC consumer. FKC is unconditional core infrastructure, so this is the ONE
/// registration path for the family: the hand-written
/// `register_with_caps_and_precision(OpKind::Rope, …, strided, …)` regs are DELETED.
///
/// The single section fans a BASE `entry_point` over `[F32, F16, F64, BF16]` (the
/// `x` + `cos` + `sin` operands share the list ⇒ they fan together, §3.4),
/// resolving `rope_<suffix>` to each distinct per-dtype wrapper and keying the
/// 4-slot `[x, cos, sin, out]` (`passthrough(x)` output), byte-for-byte the deleted
/// regs. STRIDE-AWARE (`strided: accepted` ⇒ `strided_input == true`) — the
/// caps-through-import proof. Precision is the conservative UNAUDITED author seed
/// (the elementwise pointwise precedent — the retired
/// `VULKAN_{FLOAT,HALF}_POINTWISE_PRECISION` consts are not re-asserted). Cost is
/// preserved because this runs BEFORE `fill_unset_cost_for_backend`.
///
/// The family declares NO fused ops. Init-boundary fail-fast: a parse/lower/link
/// failure of the embedded, authored contract is a programmer error surfaced once
/// here via `expect`.
fn register_vulkan_rope_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(VULKAN_ROPE_CONTRACT, &crate::fkc::VulkanLinkRegistry)
            .expect(
                "authored Vulkan rope contract must import \
                 (embedded via include_str!, resolved through VulkanLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "vulkan rope contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider
        .register_into(table, &mut fused)
        .expect("Vulkan rope contract must register into the binding table");
}

/// The authored Vulkan reduce-primitives kernel contract (production per-(op,
/// dtype) bindings), embedded via `include_str!`. Parsed + lowered by
/// [`register_vulkan_reduce_from_contract`].
const VULKAN_REDUCE_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/vulkan/reduce-prims.fkc.md");

/// Register the Vulkan reduce family (24 (op, dtype) bindings — SumReduce /
/// MaxReduce / MinReduce / MeanReduce (16) + ArgMaxDim / ArgMinDim (8)) by
/// IMPORTING its FKC kernel contract — the TWELFTH Vulkan-backend FKC consumer.
/// FKC is unconditional core infrastructure, so this is the ONE registration path
/// for the family: the hand-written `register_with_precision(OpKind::{SumReduce,…,
/// ArgMinDim}, …)` regs are DELETED.
///
/// This production contract (`reduce-prims.fkc.md`) supersedes the aspirational
/// `vulkan/reduce.fkc.md` corpus (an op-id-selector binding model FKC's
/// one-`op_kind`-per-section importer cannot express): it authors ONE section per
/// OpKind, fanned over `[F32, F16, BF16, F64]` to the DISTINCT per-dtype wrapper
/// (`reduce::{sum,max,min,mean}_*`, `arg_reduce::{argmax,argmin}_*`), keying value
/// reduces `[T, T]` (`passthrough`) and index reduces `[T, U32]` (`fixed(U32)`),
/// byte-for-byte the deleted regs. Caps ride through contiguous-only
/// (`requires_contiguous` ⇒ `strided_input == false`). Precision: value reduces are
/// the audited nondeterministic `none(reason)` seed (subgroup-tree FADD order, the
/// matmul/conv precedent); arg reduces are bitwise exact (integer index). Cost is
/// preserved because this runs BEFORE `fill_unset_cost_for_backend`.
///
/// The family declares NO fused ops. Init-boundary fail-fast: a parse/lower/link
/// failure of the embedded, authored contract is a programmer error surfaced once
/// here via `expect`.
fn register_vulkan_reduce_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(VULKAN_REDUCE_CONTRACT, &crate::fkc::VulkanLinkRegistry)
            .expect(
                "authored Vulkan reduce contract must import \
                 (embedded via include_str!, resolved through VulkanLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "vulkan reduce contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider
        .register_into(table, &mut fused)
        .expect("Vulkan reduce contract must register into the binding table");
}

/// The authored Vulkan normalization / softmax primitives kernel contract,
/// embedded via `include_str!`. Parsed + lowered by
/// [`register_vulkan_norm_from_contract`].
const VULKAN_NORM_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/vulkan/norm.fkc.md");

/// Register the Vulkan norm family (20 (op, dtype) bindings — SoftmaxLastDim (4) +
/// SoftmaxLastDimBackward (4) + LayerNormLastDim (4) + LayerNormLastDimBackward (4)
/// + RmsNormLastDim (4)) by IMPORTING its FKC kernel contract — the THIRTEENTH
/// Vulkan-backend FKC consumer. FKC is unconditional core infrastructure, so this
/// is the ONE registration path for the family: the hand-written
/// `register_with_precision(OpKind::{SoftmaxLastDim,…,RmsNormLastDim}, …)` regs are
/// DELETED.
///
/// Each section fans a BASE `entry_point` over `[F32, F16, BF16, F64]` to its
/// DISTINCT per-dtype wrapper (`softmax::*`, `norm::*`), keying forward families
/// `[T, T]` and backward families `[T, T, T]` (both inputs sharing the fanned list),
/// byte-for-byte the deleted regs. Caps ride through contiguous-only
/// (`requires_contiguous` ⇒ `strided_input == false`); precision is the audited
/// nondeterministic `none(reason)` seed (per-row subgroup-tree FADD order, the
/// matmul/conv/value-reduce precedent). Cost is preserved because this runs BEFORE
/// `fill_unset_cost_for_backend`.
///
/// The family declares NO fused ops (the FUSED softmax/norm decompositions register
/// separately via the fused registry). Init-boundary fail-fast: a parse/lower/link
/// failure of the embedded, authored contract is a programmer error surfaced once
/// here via `expect`.
fn register_vulkan_norm_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(VULKAN_NORM_CONTRACT, &crate::fkc::VulkanLinkRegistry)
            .expect(
                "authored Vulkan norm contract must import \
                 (embedded via include_str!, resolved through VulkanLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "vulkan norm contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider
        .register_into(table, &mut fused)
        .expect("Vulkan norm contract must register into the binding table");
}

/// The authored Vulkan FlashAttention kernel contract (6 sections — forward
/// f32/bf16/f16 + backward Q/K/V f32, each fanning a no-alibi + a with-alibi
/// key ⇒ 12 bindings), embedded via `include_str!` (the PRODUCTION contract).
/// Parsed + lowered by [`register_vulkan_attention_from_contract`].
const VULKAN_ATTENTION_CONTRACT: &str =
    include_str!("../../docs/kernel-contracts/vulkan/attention.fkc.md");

/// Register the Vulkan FlashAttention family (12 bindings — forward
/// `OpKind::FlashAttn` at f32/bf16/f16 + backward `OpKind::FlashAttnBackward{Q,K,V}`
/// at f32, each with a no-alibi AND a with-alibi binding shape) by IMPORTING its
/// AUTHORED FKC kernel contract — the FIFTEENTH (and LAST) Vulkan-backend FKC
/// consumer. FKC is unconditional core infrastructure, so this is the ONE
/// registration path for the family: the hand-written
/// `table.register_with_precision(OpKind::{FlashAttn,FlashAttnBackwardQ,
/// FlashAttnBackwardK,FlashAttnBackwardV}, …)` regs are DELETED. This was the
/// LAST hand-written `table.register*` call in `register_vulkan_kernels`, so the
/// Vulkan backend is now 100% contract-sourced.
///
/// The corpus's `vulkan/conv-attn-rope.fkc.md` models this family as aspirational
/// `fused_op: FLASH_ATTN` / `FLASH_ATTN_BACKWARD_*` decompositions (plus the tiled
/// FA-2 `flash_attention`), a SEPARATE future FUSED-registry seam that does NOT
/// register the primitive `OpKind::FlashAttn*` bindings production wires (the same
/// primitive-vs-fused split conv2d / matmul draw). So the migration authors SIX
/// faithful single-dtype-per-operand `op_kind:` sections in the new production
/// `docs/kernel-contracts/vulkan/attention.fkc.md`:
///  - **Forward wrappers are DISTINCT per dtype** (`flash_attn::flash_attn_{f32,
///    bf16,f16}`), so — like conv2d's `conv2d_{f32,bf16,f16}` and the CPU attention
///    contract's per-dtype `flash_attn_*` sections — each dtype is its OWN
///    single-dtype section resolving its `entry_point` AS-IS (no dtype-fan
///    umbrella); backward is three f32-only sections.
///  - **`alibi_slopes` is an `optional: true` LAST input**, so the importer's
///    optional-last fan (§3.4) keys EACH section TWICE (with + without the
///    operand), byte-for-byte the deleted regs' dual `[q,k,v,out]` /
///    `[q,k,v,alibi,out]` (forward) and `[q,k,v,do,out]` / `[q,k,v,do,alibi,out]`
///    (backward) shapes. The forward + all three backward selectors share the ONE
///    `OpParams::FlashAttn` carrier (`op_params.variant: FlashAttn` — the CPU
///    precedent; no dedicated backward variant).
///
/// **Caps ride through truthfully.** Each section's `requires_contiguous` layout
/// projects `strided_input == false` — byte-for-byte the deleted plain
/// `register_with_precision` regs (the kernels read canonical row-major q/k/v/do;
/// a strided operand is auto-Contiguized by the planner first). Precision becomes
/// the contract's corrected audited seed (`bit_stable_on_same_hardware: false`,
/// audited `none(reason)`, `determinism: nondeterministic`): the naive single-pass
/// kernel does its softmax over the `[Sk]` score row with a per-`(b,h,q)` workgroup
/// shared-memory reduction whose FADD order is scheduler-dependent, so the retired
/// hand-written `VULKAN_FLOAT_POINTWISE_PRECISION` / `VULKAN_HALF_POINTWISE_PRECISION`
/// consts (which mis-declared `bit_stable_on_same_hardware: true` + `max_ulp: 1`,
/// a per-thread-pointwise claim) OVER-CLAIMED bit-stability — the aspirational
/// `conv-attn-rope.fkc.md` already carries this honest posture; the consts are
/// retired from this seam (their `pub const` defs stay in `fused.rs`). Cost is
/// preserved because this runs BEFORE `fill_unset_cost_for_backend`, which upgrades
/// the imported `unknown_cost` sentinels to the same OpKind cost fn.
///
/// The family declares NO fused ops (the FUSED flash decompositions register
/// separately via the fused registry). Init-boundary fail-fast: a parse/lower/link
/// failure of the embedded, authored contract is a programmer error surfaced once
/// here via `expect`.
fn register_vulkan_attention_from_contract(table: &mut KernelBindingTable) {
    let provider =
        crate::fkc::import_bundle_str(VULKAN_ATTENTION_CONTRACT, &crate::fkc::VulkanLinkRegistry)
            .expect(
                "authored Vulkan attention contract must import \
                 (embedded via include_str!, resolved through VulkanLinkRegistry)",
            );
    debug_assert!(
        provider.fused.is_empty(),
        "vulkan attention contract declares no fused ops",
    );
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider
        .register_into(table, &mut fused)
        .expect("Vulkan attention contract must register into the binding table");
}

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
    // NB: the `use crate::fused::{ PrecisionGuarantee, VULKAN_FLOAT_POINTWISE_PRECISION,
    // VULKAN_HALF_POINTWISE_PRECISION }` seam is RETIRED — the FlashAttention family was
    // their LAST code user, and it now registers from the FKC attention contract (see
    // `register_vulkan_attention_from_contract`), which carries the corrected
    // audited-nondeterministic per-section precision (the two POINTWISE consts
    // over-claimed `bit_stable_on_same_hardware: true` + `max_ulp: 1` for a shared-mem
    // softmax reduction). With the FlashAttn regs gone there are NO hand-written
    // `table.register*` calls left, so NO `PrecisionGuarantee` / precision-const import
    // is needed. The `pub const` defs stay in `fused.rs`, mirroring VULKAN_TRANSCENDENTAL.
    // (VULKAN_BYTE_LEVEL_PRECISION / VULKAN_QMATMUL_PRECISION / VULKAN_TRANSCENDENTAL_PRECISION
    // / VULKAN_MATMUL_PRECISION / VULKAN_MATMUL_TENSORCORE_PRECISION / VULKAN_CAST_PRECISION
    // were retired by the earlier family migrations; KernelCaps likewise — every strided
    // Vulkan family projects `strided_input` from its FKC contract's layout.)
    let vk = BackendId::Vulkan;
    // NB: EVERY Vulkan family now registers from its FKC contract (the
    // `register_vulkan_*_from_contract` calls at the END of this fn are the SOLE
    // registration path). There are ZERO hand-written `table.register*` calls remaining
    // — the Vulkan backend is 100% contract-sourced. `vk` is used only by the
    // `fill_unset_cost_for_backend` pass below.

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

    // ----- Binary + Unary f32 (V.2.A/B) — MIGRATED to FKC contract import.
    // The hand-written `register_with_caps_and_precision` regs (6 binary + 16
    // unary, all strided f32) are DELETED; the whole elementwise family now
    // registers from `docs/kernel-contracts/vulkan/elementwise.fkc.md` via
    // `register_vulkan_elementwise_from_contract` (the SOLE path, called at the
    // END of this fn before the cost-fill pass). The top-level `strided`
    // `KernelCaps` binding that Rope / Concat / Flip / Roll / CumSum used to consume
    // is retired now those families all register from their FKC contracts
    // (movement / shape / rope), which project `strided_input` from their
    // `strided: accepted` layouts. -----

    // ----- Softmax + RmsNorm last-dim (V.2.C, f32) — per-row reductions
    // with subgroup-tree internal accumulation. CONTIGUOUS-ONLY by
    // design: the kernel issues one workgroup per row and reads the
    // row's elements flat; an arbitrary stride on the last-dim axis
    // would break the workgroup-shared-memory reduction. Auto-
    // Contiguize materializes broadcast/transpose views first. -----
    // ----- Softmax + RmsNorm + LayerNorm (+ softmax/layernorm backward) last-dim —
    // MIGRATED to FKC contract import. The hand-written `register_with_precision(
    // OpKind::{SoftmaxLastDim,SoftmaxLastDimBackward,LayerNormLastDim,
    // LayerNormLastDimBackward,RmsNormLastDim}, …)` regs (20 per-(op, dtype)
    // bindings, all contiguous) are DELETED; the whole norm family now registers
    // from `docs/kernel-contracts/vulkan/norm.fkc.md` via
    // `register_vulkan_norm_from_contract` (called at the END of this fn, before the
    // cost-fill pass). Forward families key [T, T], backward families [T, T, T];
    // precision becomes the audited nondeterministic none(reason) seed (per-row
    // subgroup-tree FADD order). Contiguous-only. -----

    // ----- SoftmaxLastDimBackward, f32 + f16/bf16/f64 (V.3.G.softmax-bwd,
    // 2026-05-30). 2 inputs (y, g) → 1 output (dx); reuses
    // OpParams::SoftmaxLastDim. Reduction order is scheduler-determined. -----
    // ----- SoftmaxLastDimBackward — MIGRATED to FKC contract import (deleted; now
    // registered from the norm contract, key [y, g, dx]). -----

    // ----- Softmax f16/bf16/f64 — MIGRATED to FKC contract import (deleted; now
    // registered from the norm contract, alongside the f32 combo). -----

    // ----- LayerNormLastDimBackward — MIGRATED to FKC contract import (deleted; now
    // registered from the norm contract, key [x, g, dx]). -----

    // ----- LayerNorm last-dim — MIGRATED to FKC contract import (deleted; now
    // registered from the norm contract). -----

    // ----- RmsNorm f16/bf16/f64 — MIGRATED to FKC contract import (deleted; now
    // registered from the norm contract, alongside the f32 combo). -----

    // ----- RoPE (V.2.C, f32) — pointwise rotation with cos/sin tables.
    // STRIDE-AWARE on `x`: rope.slang's Params struct carries
    // `x_s0/x_s1/x_s_seq/x_s_hd` + an `x_contiguous` fast-path flag
    // and decomposes per-thread index into per-dim coordinates. cos/sin
    // are documented as always-contiguous (the wrapper enforces).
    // Binding-shape MUST match the CPU registration `[x, cos, sin, out]`
    // (4 dtypes). The strided_input cap signals "ANY input may be non-
    // contiguous" — for Rope specifically, only x; the wrapper handles
    // forcing cos/sin contiguous through its own path. -----
    // ----- RoPE (f32/f16/f64/bf16) — MIGRATED to FKC contract import. The
    // hand-written `register_with_caps_and_precision(OpKind::Rope, …, strided, …)`
    // regs (4-slot [x, cos, sin, out] key, 4 per-dtype wrappers) are DELETED; the
    // rope family now registers from `docs/kernel-contracts/vulkan/rope.fkc.md` via
    // `register_vulkan_rope_from_contract` (called at the END of this fn, before the
    // cost-fill pass). Stride-aware (strided_input=true); precision becomes the
    // conservative UNAUDITED author seed (the elementwise pointwise precedent; the
    // retired VULKAN_{FLOAT,HALF}_POINTWISE_PRECISION consts are not re-asserted). -----

    // ----- IndexSelect (V.2.D, f32 src + u32 ids) — pure gather (byte-level).
    // strided-input candidate: index_select.slang flattens input to
    // (outer, axis, inner) and reads via own-shape strides. Could be
    // extended to walk arbitrary input layout strides at gather time
    // (saves an auto-Contiguize when the source is a transpose view).
    // Indices are u32; their layout-strided variant is a follow-up. -----
    // ----- IndexSelect (f32/f16/bf16/f64 src + u32 ids) — MIGRATED to FKC
    // contract import. The hand-written `register_with_precision(OpKind::
    // IndexSelect, …)` regs (4 per-dtype wrappers) are DELETED; the whole select
    // family (IndexSelect + Gather + MaskedFill) now registers from
    // `docs/kernel-contracts/vulkan/select.fkc.md` via
    // `register_vulkan_select_from_contract` (called at the END of this fn, before
    // the cost-fill pass). Caps stay contiguous-only (strided_input=false). -----

    // ----- Reduce f32 (V.2.D + V.3.A.2) — Sum / Max / Min / Mean.
    // CONTIGUOUS-ONLY by design: reduce.slang does a tree reduction
    // into workgroup-shared memory; the input is read flat. Strided
    // inputs would require either pre-materialization (current
    // behaviour via auto-Contiguize) or a redesigned reduction
    // schedule that's unlikely to outperform the auto-Contiguize +
    // current kernel. -----
    // ----- Reduce (Sum/Max/Min/Mean) + ArgMax/ArgMin — MIGRATED to FKC contract
    // import. The hand-written per-(op, dtype) `register_with_precision(OpKind::
    // {SumReduce,…,ArgMinDim}, …)` regs (16 value + 8 arg, all contiguous) are
    // DELETED; the whole reduce family now registers from
    // `docs/kernel-contracts/vulkan/reduce-prims.fkc.md` via
    // `register_vulkan_reduce_from_contract` (called at the END of this fn, before
    // the cost-fill pass). Value reduces carry the audited nondeterministic
    // none(reason) seed (subgroup-tree FADD order); arg reduces are bitwise exact
    // (integer index). Contiguous-only. -----

    // ----- IndexAdd + ScatterAdd (atomic bounded-CAS accumulate) — MIGRATED to
    // FKC contract import. The hand-written `register_with_precision(OpKind::
    // {IndexAdd,ScatterAdd}, …)` regs (4 per-dtype wrappers each, 4-slot
    // [base, U32, src, out] key) are DELETED; the whole scatter family now
    // registers from `docs/kernel-contracts/vulkan/scatter.fkc.md` via
    // `register_vulkan_scatter_from_contract` (called at the END of this fn,
    // before the cost-fill pass). Caps stay contiguous-only (strided_input=false);
    // precision becomes the contract's audited nondeterministic none(reason) seed
    // (the same posture as the deleted PrecisionGuarantee::none regs). -----

    // ----- ArgMaxDim / ArgMinDim — MIGRATED to FKC contract import (deleted; now
    // registered from the reduce-prims contract, key [T, U32] with a fixed(U32)
    // index output; bitwise exact). -----

    // ----- Reduce last-dim f16/bf16/f64 — MIGRATED to FKC contract import (deleted;
    // now registered from the reduce-prims contract, alongside the f32 combos). -----

    // ----- Concat (f32/f16/bf16/f64) — MIGRATED to FKC contract import (deleted;
    // now registered from `docs/kernel-contracts/vulkan/movement.fkc.md` via
    // `register_vulkan_movement_from_contract`, alongside CumSum). Stride-aware
    // (strided_input=true) byte-level data movement. -----

    // ----- Pad with constant fill (V.3.G.pad, 2026-05-30).
    // Byte-level kernels (b1/b2/b4/b8) dispatched by the OUTPUT dtype's
    // size at the shim. Reflect/replicate modes fall through to CPU.
    // ----- Pad + PadBackward — MIGRATED to FKC contract import. The hand-written
    // per-dtype (6 dtypes) `register_with_precision(OpKind::{Pad,PadBackward}, …)`
    // regs are DELETED; the whole pad-copy family (Pad + PadBackward + Copy) now
    // registers from `docs/kernel-contracts/vulkan/pad-copy.fkc.md` via
    // `register_vulkan_pad_copy_from_contract` (called at the END of this fn,
    // before the cost-fill pass). Each op is ONE dtype-agnostic wrapper
    // (synthetic-base umbrella); const mode only; contiguous-only. -----
    {
        // ----- MaskedFill + Gather — MIGRATED to FKC contract import (deleted;
        // now registered from `docs/kernel-contracts/vulkan/select.fkc.md` via
        // `register_vulkan_select_from_contract`, alongside IndexSelect). Both are
        // ONE dtype-agnostic wrapper across their 6 dtype keys (byte-width from the
        // output dtype) — a synthetic-base umbrella. Contiguous-only. -----
    }
    // V.3.G.concat f16/bf16/f64 — MIGRATED to FKC contract import (deleted; now
    // registered from the movement contract, alongside the f32 combo above). -----

    // ----- MatMul (f32 GEMM + the five mixed-precision / tensor-core combos)
    // — MIGRATED to FKC contract import. The hand-written
    // `table.register_with_precision(OpKind::MatMul, …)` regs (f32, f32×bf16→f32,
    // bf16×bf16→{f32,bf16}, f16×f16→{f16,f32}) are DELETED; they now register from
    // the vulkan matmul contract via `register_vulkan_matmul_from_contract`
    // (called at the END of this fn, before the cost fill pass). Caps stay
    // contiguous-only (`requires_contiguous` ⇒ strided_input=false); precision
    // becomes the contract's audited-none(reason) seed. The
    // VULKAN_MATMUL_PRECISION / VULKAN_MATMUL_TENSORCORE_PRECISION consts the
    // matmul migration left in use for the Vulkan conv2d family are now ALSO
    // retired — conv2d migrated to its own FKC contract
    // (`register_vulkan_conv_from_contract`). See
    // `docs/kernel-contracts/vulkan/matmul.fkc.md`.

    // ----- Affine / Clamp / PowI — MIGRATED to FKC contract import.
    // The hand-written Affine (f32/f16/f64 strided + bf16 contiguous-only),
    // ClampElementwise (f32), and PowIElementwise (f32) regs are DELETED; they now
    // register from the elementwise contract via
    // `register_vulkan_elementwise_from_contract` (called at the END of this fn).
    // Affine's bf16 variant keeps its contiguous-only caps
    // (`strided_input=false`) via the `affine_bf16` section's
    // `contiguous: required` layout. -----

    // ----- Cast (all pairs) — MIGRATED to FKC contract import.
    // The hand-written `table.register_with_precision(OpKind::Cast, …)`
    // regs (f32↔f16, f32↔bf16, f32↔f64, and — further below — the six
    // F8E4M3↔{f32,f16,bf16} pairs) are DELETED. The whole family now
    // registers from `docs/kernel-contracts/vulkan/cast.fkc.md` via
    // `register_vulkan_cast_from_contract`, called at the END of this fn
    // (before the cost-fill pass), the SOLE registration path.
    // The top-level `f16` / `bf16_d` bindings that the WriteSlice / Copy byte-width
    // fan-outs used to consume are retired with those families' migration to the
    // FKC pad-copy + write-slice contracts (the remaining half-precision users —
    // reduce / norm / flash-attn / rope — declare their own block-scoped locals). -----

    // ----- Binary + Unary f16 — MIGRATED to FKC contract import (deleted; now
    // registered from the elementwise contract, all strided). -----

    // ----- Binary + Unary f64 — MIGRATED to FKC contract import (deleted; now
    // registered from the elementwise contract, all strided). The top-level
    // `f64_d` binding that CumSum used to consume is retired with CumSum's
    // migration to the FKC movement contract. -----

    // ----- WriteSlice (V.3.J) — byte-width-keyed (b1/b2/b4/b8). Pure
    // byte-level data movement; no FP math.
    // CONTIGUOUS-ONLY: write_slice_b*.slang reads `src` contiguously
    // in its own rank-N shape and writes the matching slab inside
    // `dst` (also contiguous in its larger rank-N shape). The
    // destination's slab geometry IS the layout — strided inputs
    // wouldn't compose with the slab-walk's own-shape strides.
    // Auto-Contiguize materializes any non-contiguous source. -----
    // ----- WriteSlice + WriteSliceRotating — MIGRATED to FKC contract import. The
    // hand-written byte-width-keyed (9 dtypes → b1/b2/b4/b8) `register_with_precision(
    // OpKind::{WriteSlice,WriteSliceRotating}, …)` regs are DELETED; the whole
    // write-slice family now registers from
    // `docs/kernel-contracts/vulkan/write-slice.fkc.md` via
    // `register_vulkan_write_slice_from_contract` (called at the END of this fn,
    // before the cost-fill pass). Each op's 9 dtype keys share 4 byte-width wrappers
    // (the cast family's shared-wrapper precedent); contiguous-only. -----

    // ----- Binary + Unary bf16 — MIGRATED to FKC contract import (deleted). The
    // binary_bf16 regs were strided (lane-masked); the unary_bf16 regs were plain
    // `register_with_precision` (contiguous-only, `strided_input=false`) — both
    // reproduced by the contract's per-section layout projection. -----

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
    // ----- Triu / Tril / Flip / Roll — MIGRATED to FKC contract import. The
    // hand-written per-dtype (7 dtypes) `register_with_precision(OpKind::
    // {Triu,Tril}, …)` + `register_with_caps_and_precision(OpKind::{Flip,Roll},
    // …, strided, …)` regs are DELETED; the whole shape family now registers from
    // `docs/kernel-contracts/vulkan/shape.fkc.md` via
    // `register_vulkan_shape_from_contract` (called at the END of this fn, before
    // the cost-fill pass). Each op is ONE dtype-agnostic wrapper (synthetic-base
    // umbrella). Caps split by op: Triu/Tril contiguous, Flip/Roll strided. -----

    // ----- CumSum (inclusive prefix sum along one axis) — MIGRATED to FKC
    // contract import (deleted; now registered from
    // `docs/kernel-contracts/vulkan/movement.fkc.md` via
    // `register_vulkan_movement_from_contract`, alongside Concat). Stride-aware
    // (strided_input=true); precision becomes the conservative UNAUDITED author
    // seed (the elementwise pointwise precedent; the retired
    // VULKAN_{FLOAT,HALF}_POINTWISE_PRECISION consts are not re-asserted). -----

    // ----- QMatMul (Q4_0 / Q4_K_M / Q8_0) — F32 × U32-quant → F32 —
    // MIGRATED to FKC contract import. The hand-written
    // `table.register_with_precision(OpKind::QMatMul, &[F32, U32, F32], …)` reg is
    // DELETED; the single binding now registers from
    // `docs/kernel-contracts/vulkan/qmatmul.fkc.md` via
    // `register_vulkan_qmatmul_from_contract` (called at the END of this fn, before
    // the cost-fill pass). ONE wrapper `qmatmul_vk` route-picks Q4_0 (fused
    // gemv/tiled) / Q4_K_M / Q8_0 (dequant-then-matmul_f32) by
    // `OpParams::QMatMul.quant_type` — the contract keys the weight slot with the
    // LOGICAL `U32` dispatch dtype (the CPU linear-quant U8→U32 reconciliation; the
    // physical GGML block-byte honesty rides `fdx.quant`). CONTIGUOUS-ONLY: the
    // quantized weight stream has a fixed block layout (per-block scale + N
    // quantized lanes); arbitrary strides on the weight buffer would break the
    // dequant kernel's block walk (`requires_contiguous` ⇒ strided_input=false).
    // Precision becomes the contract's corrected nondeterministic seed (the retired
    // `VULKAN_QMATMUL_PRECISION` const over-claimed bit-stability, like the retired
    // `VULKAN_MATMUL_PRECISION`; the const def stays in `fused.rs`). -----

    // ----- Conv2D (f32 + bf16/f16 cooperative-matrix) — MIGRATED to FKC
    // contract import. The hand-written `table.register_with_precision(
    // OpKind::Conv2D, …)` regs (f32 here + the bf16/f16 coop combos further
    // below) are DELETED; the whole family now registers from
    // `docs/kernel-contracts/vulkan/conv.fkc.md` via
    // `register_vulkan_conv_from_contract` (called at the END of this fn, before
    // the cost-fill pass). Caps stay contiguous-only (`requires_contiguous` ⇒
    // strided_input=false); no bias key (the wrappers bail on a 3-input call);
    // precision becomes the contract's corrected nondeterministic seed (the
    // retired VULKAN_MATMUL_PRECISION / VULKAN_MATMUL_TENSORCORE_PRECISION consts
    // over-claimed bit-stability). -----

    // ----- FlashAttn (forward f32/bf16/f16) + FlashAttnBackward{Q,K,V} (f32)
    // — MIGRATED to FKC contract import. The hand-written
    // `table.register_with_precision(OpKind::{FlashAttn,FlashAttnBackwardQ,
    // FlashAttnBackwardK,FlashAttnBackwardV}, …)` regs (naive single-pass SDPA;
    // both the no-alibi [q,k,v,(do,)out] and with-alibi [q,k,v,(do,)alibi,out]
    // binding shapes) are DELETED; the whole FlashAttention family now registers
    // from `docs/kernel-contracts/vulkan/attention.fkc.md` via
    // `register_vulkan_attention_from_contract` (called at the END of this fn,
    // before the cost-fill pass). Forward is THREE distinct per-dtype wrappers
    // (per-dtype sections, the conv2d + CPU-attention precedent); backward is
    // f32-only. `alibi_slopes` is an optional-LAST input (dual-key fan);
    // op_params.variant: FlashAttn shared across forward + backward. Caps stay
    // contiguous-only (`requires_contiguous` ⇒ strided_input=false); precision
    // becomes the contract's corrected audited-nondeterministic none(reason) seed
    // (single-pass shared-mem softmax reduction is scheduler-dependent — the
    // retired VULKAN_{FLOAT,HALF}_POINTWISE_PRECISION consts over-claimed
    // bit-stability). This was the LAST hand-written table.register* call — the
    // Vulkan backend is now 100% contract-sourced. -----

    // ----- Conv2D bf16 + f16 (V.3.I extended) — im2col + cooperative-matrix
    // GEMM — MIGRATED to FKC contract import (deleted; now registered from
    // `docs/kernel-contracts/vulkan/conv.fkc.md` via
    // `register_vulkan_conv_from_contract`, alongside the f32 combo above). -----

    // ----- Cast F8E4M3 ↔ {F32, F16, BF16} — MIGRATED to FKC contract
    // import (`register_vulkan_cast_from_contract`, called below). The six
    // hand-written `OpKind::Cast` F8E4M3 regs are DELETED — the contract is
    // now the sole registration path for the whole cast family. -----

    // ----- Op::Copy D2H (bridge-retirement Phase 2). Byte-level.
    // CONTIGUOUS-ONLY: the wrapper downloads the source's bytes via
    // a Vulkan staging buffer (vkCmdCopyBuffer). The staging buffer
    // is sized to the source's flat byte count; stride-aware D2H
    // would require either per-row staging or an explicit pre-
    // Contiguize step. Auto-Contiguize handles non-contiguous
    // sources upstream of this kernel. -----
    // Op::Copy (D2H) — MIGRATED to FKC contract import (deleted; now registered
    // from `docs/kernel-contracts/vulkan/pad-copy.fkc.md` via
    // `register_vulkan_pad_copy_from_contract`, alongside Pad + PadBackward). ONE
    // dtype-agnostic wrapper (`copy_to_cpu_vulkan`) across 9 dtype keys
    // (synthetic-base umbrella); contiguous-only. -----

    // ----- Cast family (all 12 SRC↔DST pairs) — registered FROM its FKC
    // kernel contract, the SOLE path (hand-written Cast regs deleted above).
    // Runs BEFORE the cost-fill pass so its imported `unknown_cost` sentinels
    // are upgraded to the same OpKind cost fn every other Vulkan primitive
    // gets. -----
    register_vulkan_cast_from_contract(table);

    // ----- Elementwise family (unary / binary / affine / clamp / powi, 94
    // (op, dtype) bindings) — registered FROM its RE-AUTHORED FKC kernel contract,
    // the SOLE path (all hand-written elementwise regs deleted above). Runs BEFORE
    // the cost-fill pass so its imported `unknown_cost` sentinels are upgraded to
    // the same OpKind cost fn every other Vulkan primitive gets. This is the first
    // caps-through-import proof: the strided sections' layout projects
    // `strided_input=true`, the bf16-unary / affine_bf16 sections' `false`. -----
    register_vulkan_elementwise_from_contract(table);

    // ----- MatMul family (f32 GEMM + the five mixed-precision / tensor-core
    // combos, 6 (MatMul, [lhs,rhs,out]) bindings) — registered FROM its
    // RE-AUTHORED FKC kernel contract, the SOLE path (all hand-written MatMul regs
    // deleted above). Runs BEFORE the cost-fill pass so its imported
    // `unknown_cost` sentinels are upgraded to the same OpKind cost fn every other
    // Vulkan primitive gets. Caps ride through contiguous-only
    // (`requires_contiguous` ⇒ strided_input=false). -----
    register_vulkan_matmul_from_contract(table);

    // ----- Conv2D family (f32 + bf16/f16 cooperative-matrix, 3 (Conv2D,
    // [x, weight, out]) bindings) — registered FROM its RE-AUTHORED FKC kernel
    // contract, the SOLE path (all hand-written Conv2D regs deleted above). Runs
    // BEFORE the cost-fill pass so its imported `unknown_cost` sentinels are
    // upgraded to the same OpKind cost fn every other Vulkan primitive gets. Caps
    // ride through contiguous-only (`requires_contiguous` ⇒ strided_input=false);
    // no bias key (the wrappers bail on a 3-input call). -----
    register_vulkan_conv_from_contract(table);

    // ----- Select family (IndexSelect (4) + Gather (6) + MaskedFill (6), 16
    // (op, dtype) bindings) — registered FROM its FKC kernel contract, the SOLE
    // path (all hand-written IndexSelect/Gather/MaskedFill regs deleted above).
    // Runs BEFORE the cost-fill pass so its imported `unknown_cost` sentinels are
    // upgraded to the same OpKind cost fn. Caps ride through contiguous-only
    // (`requires_contiguous` ⇒ strided_input=false); Gather/MaskedFill are a
    // synthetic-base umbrella (one wrapper per op across its dtype keys). -----
    register_vulkan_select_from_contract(table);

    // ----- Scatter family (IndexAdd (4) + ScatterAdd (4), 8 (op, dtype) bindings)
    // — registered FROM its FKC kernel contract, the SOLE path (all hand-written
    // IndexAdd/ScatterAdd regs deleted above). Runs BEFORE the cost-fill pass.
    // Caps ride through contiguous-only; precision is the audited nondeterministic
    // none(reason) seed (bounded-CAS atomic accumulate). -----
    register_vulkan_scatter_from_contract(table);

    // ----- Movement family (Concat (4) + CumSum (4), 8 (op, dtype) bindings) —
    // registered FROM its FKC kernel contract, the SOLE path (all hand-written
    // Concat/CumSum regs deleted above). Runs BEFORE the cost-fill pass. Both are
    // stride-aware (`strided: accepted` ⇒ strided_input=true); Concat byte-exact,
    // CumSum conservative UNAUDITED (the elementwise pointwise precedent). -----
    register_vulkan_movement_from_contract(table);

    // ----- Shape family (Triu (7) + Tril (7) + Flip (7) + Roll (7), 28 (op, dtype)
    // bindings) — registered FROM its FKC kernel contract, the SOLE path (all
    // hand-written Triu/Tril/Flip/Roll regs deleted above). Runs BEFORE the
    // cost-fill pass. Each op is a synthetic-base umbrella (one dtype-agnostic
    // wrapper); Triu/Tril contiguous, Flip/Roll stride-aware. -----
    register_vulkan_shape_from_contract(table);

    // ----- Pad-copy family (Pad (6) + PadBackward (6) + Copy (9), 21 (op, dtype)
    // bindings) — registered FROM its FKC kernel contract, the SOLE path (all
    // hand-written Pad/PadBackward/Copy regs deleted above). Runs BEFORE the
    // cost-fill pass. Each op is a synthetic-base umbrella (one dtype-agnostic
    // wrapper); const mode only; contiguous-only. -----
    register_vulkan_pad_copy_from_contract(table);

    // ----- Write-slice family (WriteSlice (9) + WriteSliceRotating (9), 18
    // (op, dtype) bindings) — registered FROM its FKC kernel contract, the SOLE
    // path (all hand-written WriteSlice/WriteSliceRotating regs deleted above). Runs
    // BEFORE the cost-fill pass. Byte-width-keyed (9 dtypes share 4 b1/b2/b4/b8
    // wrappers per op); contiguous-only. -----
    register_vulkan_write_slice_from_contract(table);

    // ----- RoPE family (4 (Rope, [x, cos, sin, out]) per-dtype bindings) —
    // registered FROM its FKC kernel contract, the SOLE path (all hand-written Rope
    // regs deleted above). Runs BEFORE the cost-fill pass. Stride-aware
    // (strided_input=true); precision is the conservative UNAUDITED author seed. -----
    register_vulkan_rope_from_contract(table);

    // ----- Reduce family (SumReduce/MaxReduce/MinReduce/MeanReduce (16) +
    // ArgMaxDim/ArgMinDim (8), 24 (op, dtype) bindings) — registered FROM its FKC
    // kernel contract, the SOLE path (all hand-written reduce/arg-reduce regs
    // deleted above). Runs BEFORE the cost-fill pass. Contiguous-only; value reduces
    // audited nondeterministic none(reason), arg reduces bitwise exact. -----
    register_vulkan_reduce_from_contract(table);

    // ----- Norm family (SoftmaxLastDim (4) + SoftmaxLastDimBackward (4) +
    // LayerNormLastDim (4) + LayerNormLastDimBackward (4) + RmsNormLastDim (4), 20
    // (op, dtype) bindings) — registered FROM its FKC kernel contract, the SOLE path
    // (all hand-written norm/softmax regs deleted above). Runs BEFORE the cost-fill
    // pass. Contiguous-only; audited nondeterministic none(reason) (per-row
    // subgroup-tree FADD order). -----
    register_vulkan_norm_from_contract(table);

    // ----- QMatMul family (ONE (QMatMul, [F32, U32, F32]) binding) — registered
    // FROM its RE-AUTHORED FKC kernel contract, the SOLE path (the hand-written
    // `register_with_precision(OpKind::QMatMul, …)` reg deleted above). Runs BEFORE
    // the cost-fill pass so its imported `unknown_cost` sentinel is upgraded to the
    // same OpKind cost fn every other Vulkan primitive gets. ONE wrapper
    // (`qmatmul::qmatmul_vk`) route-picks Q4_0 / Q4_K_M / Q8_0 by
    // `OpParams::QMatMul.quant_type`; weight slot keyed with the LOGICAL U32 dtype
    // (the CPU linear-quant U8→U32 reconciliation; physical GGML block-byte honesty
    // on `fdx.quant`). Contiguous-only (`requires_contiguous` ⇒ strided_input=false);
    // precision is the audited nondeterministic none(reason) seed (the retired
    // VULKAN_QMATMUL_PRECISION over-claimed bit-stability). -----
    register_vulkan_qmatmul_from_contract(table);

    // ----- FlashAttention family (forward FlashAttn f32/bf16/f16 + backward
    // FlashAttnBackward{Q,K,V} f32, 12 bindings) — registered FROM its AUTHORED
    // FKC kernel contract, the SOLE path (the LAST hand-written table.register*
    // calls, all deleted above). Runs BEFORE the cost-fill pass so its imported
    // `unknown_cost` sentinels are upgraded to the same OpKind cost fn every other
    // Vulkan primitive gets. Forward is three distinct per-dtype wrappers
    // (per-dtype sections); backward is f32-only; `alibi_slopes` optional-LAST
    // (dual-key fan). Contiguous-only (`requires_contiguous` ⇒ strided_input=false);
    // precision is the corrected audited-nondeterministic none(reason) seed
    // (single-pass shared-mem softmax reduction). With this call the Vulkan backend
    // is 100% contract-sourced: `register_vulkan_kernels` contains ZERO hand-written
    // `table.register*` calls. -----
    register_vulkan_attention_from_contract(table);

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

// ===========================================================================
// FKC contract-migration tests (born-red gate for the Vulkan cast family)
// ===========================================================================
//
// The whole file is `#![cfg(feature = "vulkan")]`, so this module is compiled
// only under `--features vulkan` — no device is touched (registration is pure
// binding-table population; `register_vulkan_kernels` never probes hardware).

#[cfg(test)]
mod cast_contract_tests {
    use super::*;
    use crate::kernel::{KernelBindingTable, KernelRef};

    /// FIRST VULKAN-BACKEND FKC CONSUMER (born-red gate). `register_vulkan_kernels`
    /// registers the whole `OpKind::Cast` family (12 (SRC, DST) pairs) FROM ITS
    /// KERNEL CONTRACT (`docs/kernel-contracts/vulkan/cast.fkc.md`) via
    /// `register_vulkan_cast_from_contract` — the sole registration path, now
    /// that the hand-written `table.register_with_precision(OpKind::Cast, …)`
    /// calls (half/f64 + F8E4M3) are DELETED.
    ///
    /// For each `(Cast, [SRC, DST], Vulkan)` key this asserts:
    ///  - the binding resolves to the EXACT production wrapper fn-pointer
    ///    (behavior-preserving execution — the SAME 3 wrappers the deleted regs
    ///    used, several pairs sharing one wrapper),
    ///  - `kernel_source == "vulkan-slang"` — the contract's provenance tag (the
    ///    deleted hand-written path stamped `""`). THIS is the discriminator
    ///    that makes the test go red without the import wired: with the
    ///    hand-written cast regs removed and the import absent, the family is
    ///    simply missing from the table (lookup finds 0 alternatives),
    ///  - caps stay contiguous-only (`strided_input == false`).
    ///
    /// Precision is contract-sourced and correctly lowered per FKC §4.8, so it
    /// is NOT asserted uniformly here: the 5 `audited: true` sections carry
    /// `bit_stable_on_same_hardware`, while the 7 `audited: false` author-seed
    /// sections lower to `PrecisionGuarantee::UNAUDITED` (bit_stable=false) —
    /// the correct "no audited claim yet" posture, which DIFFERS from the
    /// retired hand-written `VULKAN_CAST_PRECISION` (which asserted bit-stable +
    /// `max_ulp: Some(0)` uniformly). The Judge audits the seeds later.
    #[test]
    fn register_vulkan_kernels_binds_cast_family_from_contract() {
        let mut table = KernelBindingTable::new();
        register_vulkan_kernels(&mut table);
        let vk = BackendId::Vulkan;

        // (SRC, DST, expected production wrapper) for all 12 cast pairs — the
        // exact wrapper each deleted hand-written reg carried.
        let cases: &[(DType, DType, KernelRef)] = &[
            (DType::F32,    DType::F16,    cast::cast_f32_half),
            (DType::F16,    DType::F32,    cast::cast_f32_half),
            (DType::F32,    DType::BF16,   cast::cast_f32_half),
            (DType::BF16,   DType::F32,    cast::cast_f32_half),
            (DType::F32,    DType::F64,    cast::cast_f32_f64),
            (DType::F64,    DType::F32,    cast::cast_f32_f64),
            (DType::F32,    DType::F8E4M3, cast_f8e4m3::cast_f8e4m3),
            (DType::F8E4M3, DType::F32,    cast_f8e4m3::cast_f8e4m3),
            (DType::F16,    DType::F8E4M3, cast_f8e4m3::cast_f8e4m3),
            (DType::F8E4M3, DType::F16,    cast_f8e4m3::cast_f8e4m3),
            (DType::BF16,   DType::F8E4M3, cast_f8e4m3::cast_f8e4m3),
            (DType::F8E4M3, DType::BF16,   cast_f8e4m3::cast_f8e4m3),
        ];

        let mut checked = 0usize;
        for (src, dst, expected) in cases {
            let alts = table.lookup_alternatives(OpKind::Cast, &[*src, *dst], vk);
            let entry = alts
                .iter()
                .find(|e| e.kernel as usize == *expected as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "Cast [{src:?}, {dst:?}]/Vulkan: the production wrapper must be bound \
                         FROM the vulkan cast contract in register_vulkan_kernels — found {} \
                         alternative(s) with sources {:?}",
                        alts.len(),
                        alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                    )
                });
            assert_eq!(
                entry.kernel_source, "vulkan-slang",
                "Cast [{src:?}, {dst:?}]: cast family must be contract-sourced \
                 (kernel_source=\"vulkan-slang\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "Cast [{src:?}, {dst:?}]: caps preserved (contiguous-only, strided_input=false)",
            );
            checked += 1;
        }
        assert_eq!(checked, 12, "all 12 (SRC, DST) cast pairs checked");
    }
}

// ===========================================================================
// FKC contract-migration tests (born-red gate for the Vulkan elementwise family)
// ===========================================================================
//
// Same `#![cfg(feature = "vulkan")]` file gate as the cast tests above: this
// module compiles only under `--features vulkan` and touches NO device
// (registration is pure binding-table population).

#[cfg(test)]
mod elementwise_contract_tests {
    use super::*;
    use crate::kernel::{KernelBindingTable, KernelRef};

    /// SECOND VULKAN-BACKEND FKC CONSUMER (born-red gate) + the FIRST
    /// caps-through-import proof. `register_vulkan_kernels` registers the whole
    /// elementwise family (94 (op, dtype) bindings) FROM ITS RE-AUTHORED KERNEL
    /// CONTRACT (`docs/kernel-contracts/vulkan/elementwise.fkc.md`) via
    /// `register_vulkan_elementwise_from_contract` -- the sole registration path,
    /// now that the hand-written per-(op,dtype) unary(16)/binary(6)/affine/clamp/
    /// powi regs are DELETED.
    ///
    /// For each `(OpKind, [T..], Vulkan)` key this asserts: the binding resolves
    /// to the EXACT production wrapper fn-pointer (behavior-preserving); the
    /// `kernel_source == "vulkan-slang"` (the RED discriminator -- the deleted
    /// hand path stamped `""`); and `caps.strided_input` MATCHES THE HAND-WRITTEN
    /// TRUTH PER KEY -- true for the strided unary(f32/f16/f64)/binary(all 4)/
    /// affine(f32/f16/f64)/clamp/powi keys, false for the contiguous-only
    /// bf16-unary / affine_bf16 keys. That last assertion is the
    /// caps-through-import proof: the contract's per-operand five-flag layout
    /// projects onto the binding's `KernelCaps.strided_input` exactly as the hand
    /// path set it. Precision is contract-sourced (`audited: false` -> UNAUDITED,
    /// the cast-migration precedent) and NOT asserted here.
    #[test]
    fn register_vulkan_kernels_binds_elementwise_family_from_contract() {
        let mut table = KernelBindingTable::new();
        register_vulkan_kernels(&mut table);
        let vk = BackendId::Vulkan;

        // (OpKind, key dtypes, expected production wrapper, expected strided_input)
        let cases: &[(OpKind, &[DType], KernelRef, bool)] = &[
        // ---- Unary: 16 ops x {f32,f16,f64 strided; bf16 contiguous} -- key [T, T]. ----
        (OpKind::NegElementwise, &[DType::F32, DType::F32], unary::neg_f32 as KernelRef, true),
        (OpKind::NegElementwise, &[DType::F16, DType::F16], unary_f16::neg_f16 as KernelRef, true),
        (OpKind::NegElementwise, &[DType::F64, DType::F64], unary_f64::neg_f64 as KernelRef, true),
        (OpKind::NegElementwise, &[DType::BF16, DType::BF16], unary_bf16::neg_bf16 as KernelRef, false),
        (OpKind::SqrElementwise, &[DType::F32, DType::F32], unary::sqr_f32 as KernelRef, true),
        (OpKind::SqrElementwise, &[DType::F16, DType::F16], unary_f16::sqr_f16 as KernelRef, true),
        (OpKind::SqrElementwise, &[DType::F64, DType::F64], unary_f64::sqr_f64 as KernelRef, true),
        (OpKind::SqrElementwise, &[DType::BF16, DType::BF16], unary_bf16::sqr_bf16 as KernelRef, false),
        (OpKind::SqrtElementwise, &[DType::F32, DType::F32], unary::sqrt_f32 as KernelRef, true),
        (OpKind::SqrtElementwise, &[DType::F16, DType::F16], unary_f16::sqrt_f16 as KernelRef, true),
        (OpKind::SqrtElementwise, &[DType::F64, DType::F64], unary_f64::sqrt_f64 as KernelRef, true),
        (OpKind::SqrtElementwise, &[DType::BF16, DType::BF16], unary_bf16::sqrt_bf16 as KernelRef, false),
        (OpKind::ExpElementwise, &[DType::F32, DType::F32], unary::exp_f32 as KernelRef, true),
        (OpKind::ExpElementwise, &[DType::F16, DType::F16], unary_f16::exp_f16 as KernelRef, true),
        (OpKind::ExpElementwise, &[DType::F64, DType::F64], unary_f64::exp_f64 as KernelRef, true),
        (OpKind::ExpElementwise, &[DType::BF16, DType::BF16], unary_bf16::exp_bf16 as KernelRef, false),
        (OpKind::LogElementwise, &[DType::F32, DType::F32], unary::log_f32 as KernelRef, true),
        (OpKind::LogElementwise, &[DType::F16, DType::F16], unary_f16::log_f16 as KernelRef, true),
        (OpKind::LogElementwise, &[DType::F64, DType::F64], unary_f64::log_f64 as KernelRef, true),
        (OpKind::LogElementwise, &[DType::BF16, DType::BF16], unary_bf16::log_bf16 as KernelRef, false),
        (OpKind::SinElementwise, &[DType::F32, DType::F32], unary::sin_f32 as KernelRef, true),
        (OpKind::SinElementwise, &[DType::F16, DType::F16], unary_f16::sin_f16 as KernelRef, true),
        (OpKind::SinElementwise, &[DType::F64, DType::F64], unary_f64::sin_f64 as KernelRef, true),
        (OpKind::SinElementwise, &[DType::BF16, DType::BF16], unary_bf16::sin_bf16 as KernelRef, false),
        (OpKind::CosElementwise, &[DType::F32, DType::F32], unary::cos_f32 as KernelRef, true),
        (OpKind::CosElementwise, &[DType::F16, DType::F16], unary_f16::cos_f16 as KernelRef, true),
        (OpKind::CosElementwise, &[DType::F64, DType::F64], unary_f64::cos_f64 as KernelRef, true),
        (OpKind::CosElementwise, &[DType::BF16, DType::BF16], unary_bf16::cos_bf16 as KernelRef, false),
        (OpKind::TanhElementwise, &[DType::F32, DType::F32], unary::tanh_f32 as KernelRef, true),
        (OpKind::TanhElementwise, &[DType::F16, DType::F16], unary_f16::tanh_f16 as KernelRef, true),
        (OpKind::TanhElementwise, &[DType::F64, DType::F64], unary_f64::tanh_f64 as KernelRef, true),
        (OpKind::TanhElementwise, &[DType::BF16, DType::BF16], unary_bf16::tanh_bf16 as KernelRef, false),
        (OpKind::SigmoidElementwise, &[DType::F32, DType::F32], unary::sigmoid_f32 as KernelRef, true),
        (OpKind::SigmoidElementwise, &[DType::F16, DType::F16], unary_f16::sigmoid_f16 as KernelRef, true),
        (OpKind::SigmoidElementwise, &[DType::F64, DType::F64], unary_f64::sigmoid_f64 as KernelRef, true),
        (OpKind::SigmoidElementwise, &[DType::BF16, DType::BF16], unary_bf16::sigmoid_bf16 as KernelRef, false),
        (OpKind::SiluElementwise, &[DType::F32, DType::F32], unary::silu_f32 as KernelRef, true),
        (OpKind::SiluElementwise, &[DType::F16, DType::F16], unary_f16::silu_f16 as KernelRef, true),
        (OpKind::SiluElementwise, &[DType::F64, DType::F64], unary_f64::silu_f64 as KernelRef, true),
        (OpKind::SiluElementwise, &[DType::BF16, DType::BF16], unary_bf16::silu_bf16 as KernelRef, false),
        (OpKind::GeluElementwise, &[DType::F32, DType::F32], unary::gelu_f32 as KernelRef, true),
        (OpKind::GeluElementwise, &[DType::F16, DType::F16], unary_f16::gelu_f16 as KernelRef, true),
        (OpKind::GeluElementwise, &[DType::F64, DType::F64], unary_f64::gelu_f64 as KernelRef, true),
        (OpKind::GeluElementwise, &[DType::BF16, DType::BF16], unary_bf16::gelu_bf16 as KernelRef, false),
        (OpKind::ReluElementwise, &[DType::F32, DType::F32], unary::relu_f32 as KernelRef, true),
        (OpKind::ReluElementwise, &[DType::F16, DType::F16], unary_f16::relu_f16 as KernelRef, true),
        (OpKind::ReluElementwise, &[DType::F64, DType::F64], unary_f64::relu_f64 as KernelRef, true),
        (OpKind::ReluElementwise, &[DType::BF16, DType::BF16], unary_bf16::relu_bf16 as KernelRef, false),
        (OpKind::StepElementwise, &[DType::F32, DType::F32], unary::step_f32 as KernelRef, true),
        (OpKind::StepElementwise, &[DType::F16, DType::F16], unary_f16::step_f16 as KernelRef, true),
        (OpKind::StepElementwise, &[DType::F64, DType::F64], unary_f64::step_f64 as KernelRef, true),
        (OpKind::StepElementwise, &[DType::BF16, DType::BF16], unary_bf16::step_bf16 as KernelRef, false),
        (OpKind::AbsElementwise, &[DType::F32, DType::F32], unary::abs_f32 as KernelRef, true),
        (OpKind::AbsElementwise, &[DType::F16, DType::F16], unary_f16::abs_f16 as KernelRef, true),
        (OpKind::AbsElementwise, &[DType::F64, DType::F64], unary_f64::abs_f64 as KernelRef, true),
        (OpKind::AbsElementwise, &[DType::BF16, DType::BF16], unary_bf16::abs_bf16 as KernelRef, false),
        (OpKind::SignElementwise, &[DType::F32, DType::F32], unary::sign_f32 as KernelRef, true),
        (OpKind::SignElementwise, &[DType::F16, DType::F16], unary_f16::sign_f16 as KernelRef, true),
        (OpKind::SignElementwise, &[DType::F64, DType::F64], unary_f64::sign_f64 as KernelRef, true),
        (OpKind::SignElementwise, &[DType::BF16, DType::BF16], unary_bf16::sign_bf16 as KernelRef, false),
        (OpKind::RecipElementwise, &[DType::F32, DType::F32], unary::recip_f32 as KernelRef, true),
        (OpKind::RecipElementwise, &[DType::F16, DType::F16], unary_f16::recip_f16 as KernelRef, true),
        (OpKind::RecipElementwise, &[DType::F64, DType::F64], unary_f64::recip_f64 as KernelRef, true),
        (OpKind::RecipElementwise, &[DType::BF16, DType::BF16], unary_bf16::recip_bf16 as KernelRef, false),
        // ---- Binary: 6 ops x {f32,f16,f64,bf16} all strided -- key [T, T, T]. ----
        (OpKind::AddElementwise, &[DType::F32, DType::F32, DType::F32], binary::add_f32 as KernelRef, true),
        (OpKind::AddElementwise, &[DType::F16, DType::F16, DType::F16], binary_f16::add_f16 as KernelRef, true),
        (OpKind::AddElementwise, &[DType::F64, DType::F64, DType::F64], binary_f64::add_f64 as KernelRef, true),
        (OpKind::AddElementwise, &[DType::BF16, DType::BF16, DType::BF16], binary_bf16::add_bf16 as KernelRef, true),
        (OpKind::SubElementwise, &[DType::F32, DType::F32, DType::F32], binary::sub_f32 as KernelRef, true),
        (OpKind::SubElementwise, &[DType::F16, DType::F16, DType::F16], binary_f16::sub_f16 as KernelRef, true),
        (OpKind::SubElementwise, &[DType::F64, DType::F64, DType::F64], binary_f64::sub_f64 as KernelRef, true),
        (OpKind::SubElementwise, &[DType::BF16, DType::BF16, DType::BF16], binary_bf16::sub_bf16 as KernelRef, true),
        (OpKind::MulElementwise, &[DType::F32, DType::F32, DType::F32], binary::mul_f32 as KernelRef, true),
        (OpKind::MulElementwise, &[DType::F16, DType::F16, DType::F16], binary_f16::mul_f16 as KernelRef, true),
        (OpKind::MulElementwise, &[DType::F64, DType::F64, DType::F64], binary_f64::mul_f64 as KernelRef, true),
        (OpKind::MulElementwise, &[DType::BF16, DType::BF16, DType::BF16], binary_bf16::mul_bf16 as KernelRef, true),
        (OpKind::DivElementwise, &[DType::F32, DType::F32, DType::F32], binary::div_f32 as KernelRef, true),
        (OpKind::DivElementwise, &[DType::F16, DType::F16, DType::F16], binary_f16::div_f16 as KernelRef, true),
        (OpKind::DivElementwise, &[DType::F64, DType::F64, DType::F64], binary_f64::div_f64 as KernelRef, true),
        (OpKind::DivElementwise, &[DType::BF16, DType::BF16, DType::BF16], binary_bf16::div_bf16 as KernelRef, true),
        (OpKind::MaximumElementwise, &[DType::F32, DType::F32, DType::F32], binary::maximum_f32 as KernelRef, true),
        (OpKind::MaximumElementwise, &[DType::F16, DType::F16, DType::F16], binary_f16::maximum_f16 as KernelRef, true),
        (OpKind::MaximumElementwise, &[DType::F64, DType::F64, DType::F64], binary_f64::maximum_f64 as KernelRef, true),
        (OpKind::MaximumElementwise, &[DType::BF16, DType::BF16, DType::BF16], binary_bf16::maximum_bf16 as KernelRef, true),
        (OpKind::MinimumElementwise, &[DType::F32, DType::F32, DType::F32], binary::minimum_f32 as KernelRef, true),
        (OpKind::MinimumElementwise, &[DType::F16, DType::F16, DType::F16], binary_f16::minimum_f16 as KernelRef, true),
        (OpKind::MinimumElementwise, &[DType::F64, DType::F64, DType::F64], binary_f64::minimum_f64 as KernelRef, true),
        (OpKind::MinimumElementwise, &[DType::BF16, DType::BF16, DType::BF16], binary_bf16::minimum_bf16 as KernelRef, true),
        // ---- Affine: f32/f16/f64 strided + bf16 contiguous -- key [T, T]. ----
        (OpKind::Affine, &[DType::F32, DType::F32], affine::affine_f32 as KernelRef, true),
        (OpKind::Affine, &[DType::F16, DType::F16], affine::affine_f16 as KernelRef, true),
        (OpKind::Affine, &[DType::F64, DType::F64], affine::affine_f64 as KernelRef, true),
        (OpKind::Affine, &[DType::BF16, DType::BF16], affine::affine_bf16 as KernelRef, false),
        // ---- Clamp / PowI: single-dtype f32 strided -- key [T, T]. ----
        (OpKind::ClampElementwise, &[DType::F32, DType::F32], clamp::clamp_f32 as KernelRef, true),
        (OpKind::PowIElementwise, &[DType::F32, DType::F32], powi::powi_f32 as KernelRef, true),
        ];

        let mut checked = 0usize;
        for (op, key, expected, expect_strided) in cases {
            let alts = table.lookup_alternatives(*op, key, vk);
            let entry = alts.iter().find(|e| e.kernel as usize == *expected as usize);
            let entry = match entry {
                Some(e) => e,
                None => panic!(
                    "{op:?} {key:?}/Vulkan: production wrapper must be bound FROM the vulkan elementwise contract; found {} alt(s) with sources {:?}",
                    alts.len(),
                    alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                ),
            };
            assert_eq!(
                entry.kernel_source, "vulkan-slang",
                "{op:?} {key:?}: elementwise family must be contract-sourced (kernel_source=vulkan-slang); got {:?}",
                entry.kernel_source,
            );
            assert_eq!(
                entry.caps.strided_input, *expect_strided,
                "{op:?} {key:?}: caps must ride through the import truthfully (strided_input matches the hand-written reg)",
            );
            checked += 1;
        }
        assert_eq!(checked, 94, "all 94 (op, dtype) elementwise bindings checked");
    }
}

// ===========================================================================
// FKC contract-migration tests (born-red gate for the Vulkan matmul family)
// ===========================================================================
//
// Same `#![cfg(feature = "vulkan")]` file gate as the cast / elementwise tests
// above: this module compiles only under `--features vulkan` and touches NO
// device (registration is pure binding-table population).

#[cfg(test)]
mod matmul_contract_tests {
    use super::*;
    use crate::kernel::{KernelBindingTable, KernelRef};

    /// THIRD VULKAN-BACKEND FKC CONSUMER (born-red gate). `register_vulkan_kernels`
    /// registers the whole Vulkan matmul family (6 per-combo `(MatMul, [lhs,rhs,out])`
    /// bindings) FROM ITS RE-AUTHORED KERNEL CONTRACT
    /// (`docs/kernel-contracts/vulkan/matmul.fkc.md`) via
    /// `register_vulkan_matmul_from_contract` — the sole registration path, now
    /// that the hand-written `table.register_with_precision(OpKind::MatMul, …)`
    /// regs (the f32 GEMM + the five mixed-precision / tensor-core combos) are
    /// DELETED.
    ///
    /// This ALSO dissolves the `dispatch/matmul.fkc.md :: matmul_mixed_precision`
    /// multi-axis `FanoutDtypeMismatch` corpus deferral: the aspirational single
    /// multi-dtype section (lhs `[F32,BF16,F16]` vs rhs `[BF16,F16]`) is replaced
    /// by EXPLICIT single-dtype-per-operand per-combo sections, exactly the cast
    /// family's per-pair precedent.
    ///
    /// For each `(MatMul, [lhs,rhs,out], Vulkan)` key this asserts: the binding
    /// resolves to the EXACT production wrapper fn-pointer (behavior-preserving);
    /// `kernel_source == "vulkan-slang"` (the RED discriminator — the deleted hand
    /// path stamped `""`, so before the import is wired this assert fails on the
    /// empty source); and `caps.strided_input == false` — the contiguous-only
    /// truth of the deleted `register_with_precision` regs (the coop / vec4 loads
    /// require canonical row-major, so a strided operand is auto-Contiguized
    /// first). Precision is contract-sourced (the cast/elementwise precedent) and
    /// NOT asserted here.
    #[test]
    fn register_vulkan_kernels_binds_matmul_family_from_contract() {
        let mut table = KernelBindingTable::new();
        register_vulkan_kernels(&mut table);
        let vk = BackendId::Vulkan;

        // (key dtypes [lhs, rhs, out], expected production wrapper). Every matmul
        // binding is contiguous-only (strided_input == false), so that is checked
        // uniformly below rather than per-row.
        let cases: &[(&[DType], KernelRef)] = &[
            (&[DType::F32,  DType::F32,  DType::F32],  matmul::matmul_f32 as KernelRef),
            (&[DType::F32,  DType::BF16, DType::F32],  matmul::matmul_f32_bf16_b as KernelRef),
            (&[DType::BF16, DType::BF16, DType::F32],  matmul::matmul_bf16_bf16_f32 as KernelRef),
            (&[DType::BF16, DType::BF16, DType::BF16], matmul::matmul_bf16_bf16_bf16 as KernelRef),
            (&[DType::F16,  DType::F16,  DType::F16],  matmul::matmul_f16_f16_f16 as KernelRef),
            (&[DType::F16,  DType::F16,  DType::F32],  matmul::matmul_f16_f16_f32 as KernelRef),
        ];

        let mut checked = 0usize;
        for (key, expected) in cases {
            let alts = table.lookup_alternatives(OpKind::MatMul, key, vk);
            let entry = alts.iter().find(|e| e.kernel as usize == *expected as usize);
            let entry = match entry {
                Some(e) => e,
                None => panic!(
                    "MatMul {key:?}/Vulkan: production wrapper must be bound FROM the vulkan \
                     matmul contract in register_vulkan_kernels; found {} alt(s) with sources {:?}",
                    alts.len(),
                    alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                ),
            };
            assert_eq!(
                entry.kernel_source, "vulkan-slang",
                "MatMul {key:?}: matmul family must be contract-sourced \
                 (kernel_source=\"vulkan-slang\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "MatMul {key:?}: caps preserved (contiguous-only, strided_input=false)",
            );
            checked += 1;
        }
        assert_eq!(checked, 6, "all 6 (MatMul, [lhs,rhs,out]) Vulkan combos checked");
    }
}

// ===========================================================================
// FKC contract-migration tests (born-red gate for the Vulkan conv2d family)
// ===========================================================================
//
// Same `#![cfg(feature = "vulkan")]` file gate as the cast/elementwise/matmul
// tests above: this module compiles only under `--features vulkan` and touches
// NO device (registration is pure binding-table population).

#[cfg(test)]
mod conv_contract_tests {
    use super::*;
    use crate::kernel::{KernelBindingTable, KernelRef};

    /// FOURTH VULKAN-BACKEND FKC CONSUMER (born-red gate). `register_vulkan_kernels`
    /// registers the whole Vulkan conv2d family (3 per-(op, dtype) `(Conv2D,
    /// [x, weight, out])` bindings) FROM ITS RE-AUTHORED KERNEL CONTRACT
    /// (`docs/kernel-contracts/vulkan/conv.fkc.md`) via
    /// `register_vulkan_conv_from_contract` — the sole registration path, now that
    /// the hand-written `table.register_with_precision(OpKind::Conv2D, …)` regs
    /// (f32 / bf16 / f16) are DELETED.
    ///
    /// The corpus's `vulkan/conv-attn-rope.fkc.md` describes conv2d as an
    /// ASPIRATIONAL `fused_op: CONV2D` **im2col STAGE** (`conv2d_im2col_f32` /
    /// `conv2d_im2col_bf16`, output = the patches matrix), which is NOT what
    /// production registers: production registers a PRIMITIVE `OpKind::Conv2D`
    /// binding whose wrapper (`conv2d::conv2d_f32` → `VulkanBackend::conv2d_*_bytes`)
    /// runs the WHOLE conv (im2col + matmul internally). This mirrors the matmul
    /// migration (aspirational `matmul_mixed_precision` chassis superseded by
    /// EXPLICIT per-combo `op_kind` sections in the production `vulkan/matmul.fkc.md`).
    ///
    /// For each `(Conv2D, [x, weight, out], Vulkan)` key this asserts: the binding
    /// resolves to the EXACT production wrapper fn-pointer (behavior-preserving);
    /// `kernel_source == "vulkan-slang"` (the RED discriminator — the deleted hand
    /// path stamped `""`, so before the import is wired this assert fails on the
    /// empty source); and `caps.strided_input == false` — the contiguous-only truth
    /// of the deleted `register_with_precision` regs (conv2d_im2col reads NCHW with
    /// canonical strides, so a strided operand is auto-Contiguized first). No bias
    /// key is registered: the Vulkan wrappers bail on a 3-input (bias) call, so the
    /// contract declares NO optional bias operand and each section keys ONLY the
    /// 3-slot `[x, weight, out]` (unlike the CPU conv contract's optional-bias dual
    /// key). Precision is contract-sourced (the cast/elementwise/matmul precedent)
    /// and NOT asserted here.
    #[test]
    fn register_vulkan_kernels_binds_conv_family_from_contract() {
        let mut table = KernelBindingTable::new();
        register_vulkan_kernels(&mut table);
        let vk = BackendId::Vulkan;

        // (key dtypes [x, weight, out], expected production wrapper). Every conv
        // binding is contiguous-only (strided_input == false), so that is checked
        // uniformly below rather than per-row.
        let cases: &[(&[DType], KernelRef)] = &[
            (&[DType::F32,  DType::F32,  DType::F32],  conv2d::conv2d_f32 as KernelRef),
            (&[DType::BF16, DType::BF16, DType::BF16], conv2d::conv2d_bf16 as KernelRef),
            (&[DType::F16,  DType::F16,  DType::F16],  conv2d::conv2d_f16 as KernelRef),
        ];

        let mut checked = 0usize;
        for (key, expected) in cases {
            let alts = table.lookup_alternatives(OpKind::Conv2D, key, vk);
            let entry = alts.iter().find(|e| e.kernel as usize == *expected as usize);
            let entry = match entry {
                Some(e) => e,
                None => panic!(
                    "Conv2D {key:?}/Vulkan: production wrapper must be bound FROM the vulkan \
                     conv contract in register_vulkan_kernels; found {} alt(s) with sources {:?}",
                    alts.len(),
                    alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                ),
            };
            assert_eq!(
                entry.kernel_source, "vulkan-slang",
                "Conv2D {key:?}: conv family must be contract-sourced \
                 (kernel_source=\"vulkan-slang\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "Conv2D {key:?}: caps preserved (contiguous-only, strided_input=false)",
            );
            checked += 1;
        }
        assert_eq!(checked, 3, "all 3 (Conv2D, [x, weight, out]) Vulkan combos checked");
    }
}

// ===========================================================================
// FKC contract-migration tests (born-red gate for the Vulkan select family)
// ===========================================================================
//
// Same `#![cfg(feature = "vulkan")]` file gate: compiles only under
// `--features vulkan` and touches NO device (registration is pure binding-table
// population).

#[cfg(test)]
mod select_contract_tests {
    use super::*;
    use crate::kernel::{KernelBindingTable, KernelRef};

    /// FIFTH VULKAN-BACKEND FKC CONSUMER (born-red gate). `register_vulkan_kernels`
    /// registers the whole Vulkan **select** family — IndexSelect (4 per-dtype
    /// wrappers), Gather (6 dtype keys → the ONE `gather::gather` synthetic-base
    /// umbrella wrapper), MaskedFill (6 dtype keys → the ONE
    /// `masked_fill::masked_fill` umbrella wrapper) — FROM ITS FKC KERNEL CONTRACT
    /// (`docs/kernel-contracts/vulkan/select.fkc.md`) via
    /// `register_vulkan_select_from_contract`, the sole registration path now that
    /// the hand-written `register_with_precision(OpKind::{IndexSelect,Gather,
    /// MaskedFill}, …)` regs are DELETED.
    ///
    /// Per key: the binding resolves to the EXACT production wrapper fn-pointer
    /// (behavior-preserving; several Gather/MaskedFill keys share ONE wrapper, the
    /// pad_cpu synthetic-umbrella precedent); `kernel_source == "vulkan-slang"`
    /// (the RED discriminator — the deleted hand path stamped `""`); and
    /// `caps.strided_input == false` — the contiguous-only truth of the deleted
    /// plain `register_with_precision` regs (byte-level data movers read/write
    /// their own-shape flat buffers; a strided operand is auto-Contiguized first).
    /// Precision is contract-sourced (byte-exact `audited: true`, the cast/matmul
    /// precedent) and NOT asserted here.
    #[test]
    fn register_vulkan_kernels_binds_select_family_from_contract() {
        let mut table = KernelBindingTable::new();
        register_vulkan_kernels(&mut table);
        let vk = BackendId::Vulkan;

        let f32 = DType::F32; let f16 = DType::F16; let bf16 = DType::BF16;
        let f64 = DType::F64; let u8 = DType::U8; let u32 = DType::U32;

        // (OpKind, key dtypes, expected production wrapper). Every select binding
        // is contiguous-only (strided_input == false), checked uniformly below.
        let cases: &[(OpKind, &[DType], KernelRef)] = &[
            // IndexSelect: [source, U32, out]; 4 distinct per-dtype wrappers.
            (OpKind::IndexSelect, &[f32,  u32, f32],  indexing::index_select_f32 as KernelRef),
            (OpKind::IndexSelect, &[f16,  u32, f16],  indexing::index_select_f16 as KernelRef),
            (OpKind::IndexSelect, &[bf16, u32, bf16], indexing::index_select_bf16 as KernelRef),
            (OpKind::IndexSelect, &[f64,  u32, f64],  indexing::index_select_f64 as KernelRef),
            // Gather: [source, U32, out]; ONE dtype-agnostic wrapper across 6 keys.
            (OpKind::Gather, &[f32,  u32, f32],  gather::gather as KernelRef),
            (OpKind::Gather, &[f16,  u32, f16],  gather::gather as KernelRef),
            (OpKind::Gather, &[bf16, u32, bf16], gather::gather as KernelRef),
            (OpKind::Gather, &[f64,  u32, f64],  gather::gather as KernelRef),
            (OpKind::Gather, &[u8,   u32, u8],   gather::gather as KernelRef),
            (OpKind::Gather, &[u32,  u32, u32],  gather::gather as KernelRef),
            // MaskedFill: [data, U8 mask, out]; ONE wrapper across 6 keys.
            (OpKind::MaskedFill, &[f32,  u8, f32],  masked_fill::masked_fill as KernelRef),
            (OpKind::MaskedFill, &[f16,  u8, f16],  masked_fill::masked_fill as KernelRef),
            (OpKind::MaskedFill, &[bf16, u8, bf16], masked_fill::masked_fill as KernelRef),
            (OpKind::MaskedFill, &[f64,  u8, f64],  masked_fill::masked_fill as KernelRef),
            (OpKind::MaskedFill, &[u8,   u8, u8],   masked_fill::masked_fill as KernelRef),
            (OpKind::MaskedFill, &[u32,  u8, u32],  masked_fill::masked_fill as KernelRef),
        ];

        let mut checked = 0usize;
        for (op, key, expected) in cases {
            let alts = table.lookup_alternatives(*op, key, vk);
            let entry = alts.iter().find(|e| e.kernel as usize == *expected as usize);
            let entry = match entry {
                Some(e) => e,
                None => panic!(
                    "{op:?} {key:?}/Vulkan: production wrapper must be bound FROM the vulkan \
                     select contract in register_vulkan_kernels; found {} alt(s) with sources {:?}",
                    alts.len(),
                    alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                ),
            };
            assert_eq!(
                entry.kernel_source, "vulkan-slang",
                "{op:?} {key:?}: select family must be contract-sourced \
                 (kernel_source=\"vulkan-slang\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "{op:?} {key:?}: caps preserved (contiguous-only, strided_input=false)",
            );
            checked += 1;
        }
        assert_eq!(checked, 16, "all 16 (IndexSelect/Gather/MaskedFill) select keys checked");
    }
}

// ===========================================================================
// FKC contract-migration tests (born-red gate for the Vulkan scatter family)
// ===========================================================================

#[cfg(test)]
mod scatter_contract_tests {
    use super::*;
    use crate::kernel::{KernelBindingTable, KernelRef};

    /// SIXTH VULKAN-BACKEND FKC CONSUMER (born-red gate). `register_vulkan_kernels`
    /// registers the whole Vulkan **scatter** family — IndexAdd (4 per-dtype) +
    /// ScatterAdd (4 per-dtype), 4-slot key `[base, U32, src, out]` — FROM ITS FKC
    /// KERNEL CONTRACT (`docs/kernel-contracts/vulkan/scatter.fkc.md`) via
    /// `register_vulkan_scatter_from_contract`, the sole path now that the
    /// hand-written `register_with_precision(OpKind::{IndexAdd,ScatterAdd}, …)` regs
    /// are DELETED. Each section fans `base` + `src` over the SAME `[F32,F64,BF16,F16]`
    /// list (§3.4; both operands share the list ⇒ one fan, not a mismatch), keying
    /// `[T, U32, T, T]`.
    ///
    /// Per key: EXACT production wrapper fn-pointer; `kernel_source == "vulkan-slang"`
    /// (RED discriminator); `caps.strided_input == false` (contiguous-only atomic
    /// scatter). Precision is contract-sourced (nondeterministic bounded-CAS,
    /// audited none(reason) — the matmul/conv precedent) and NOT asserted here.
    #[test]
    fn register_vulkan_kernels_binds_scatter_family_from_contract() {
        let mut table = KernelBindingTable::new();
        register_vulkan_kernels(&mut table);
        let vk = BackendId::Vulkan;

        let f32 = DType::F32; let f16 = DType::F16; let bf16 = DType::BF16;
        let f64 = DType::F64; let u32 = DType::U32;

        let cases: &[(OpKind, &[DType], KernelRef)] = &[
            (OpKind::IndexAdd, &[f32,  u32, f32,  f32],  index_add::index_add_f32 as KernelRef),
            (OpKind::IndexAdd, &[f64,  u32, f64,  f64],  index_add::index_add_f64 as KernelRef),
            (OpKind::IndexAdd, &[bf16, u32, bf16, bf16], index_add::index_add_bf16 as KernelRef),
            (OpKind::IndexAdd, &[f16,  u32, f16,  f16],  index_add::index_add_f16 as KernelRef),
            (OpKind::ScatterAdd, &[f32,  u32, f32,  f32],  scatter_add::scatter_add_f32 as KernelRef),
            (OpKind::ScatterAdd, &[f64,  u32, f64,  f64],  scatter_add::scatter_add_f64 as KernelRef),
            (OpKind::ScatterAdd, &[bf16, u32, bf16, bf16], scatter_add::scatter_add_bf16 as KernelRef),
            (OpKind::ScatterAdd, &[f16,  u32, f16,  f16],  scatter_add::scatter_add_f16 as KernelRef),
        ];

        let mut checked = 0usize;
        for (op, key, expected) in cases {
            let alts = table.lookup_alternatives(*op, key, vk);
            let entry = alts.iter().find(|e| e.kernel as usize == *expected as usize);
            let entry = match entry {
                Some(e) => e,
                None => panic!(
                    "{op:?} {key:?}/Vulkan: production wrapper must be bound FROM the vulkan \
                     scatter contract in register_vulkan_kernels; found {} alt(s) with sources {:?}",
                    alts.len(),
                    alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                ),
            };
            assert_eq!(
                entry.kernel_source, "vulkan-slang",
                "{op:?} {key:?}: scatter family must be contract-sourced \
                 (kernel_source=\"vulkan-slang\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "{op:?} {key:?}: caps preserved (contiguous-only, strided_input=false)",
            );
            checked += 1;
        }
        assert_eq!(checked, 8, "all 8 (IndexAdd/ScatterAdd) scatter keys checked");
    }
}

// ===========================================================================
// FKC contract-migration tests (born-red gate for the Vulkan movement family)
// ===========================================================================

#[cfg(test)]
mod movement_contract_tests {
    use super::*;
    use crate::kernel::{KernelBindingTable, KernelRef};

    /// SEVENTH VULKAN-BACKEND FKC CONSUMER (born-red gate). `register_vulkan_kernels`
    /// registers the whole Vulkan **movement** family — Concat (4 per-dtype) +
    /// CumSum (4 per-dtype), key `[T, T]` — FROM ITS FKC KERNEL CONTRACT
    /// (`docs/kernel-contracts/vulkan/movement.fkc.md`) via
    /// `register_vulkan_movement_from_contract`, the sole path now that the
    /// hand-written `register_with_caps_and_precision(OpKind::{Concat,CumSum}, …,
    /// strided, …)` regs are DELETED.
    ///
    /// Both are STRIDE-AWARE (the deleted regs carried `strided`), so
    /// `caps.strided_input == true` is the caps-through-import proof here (the
    /// contract's `strided: accepted` layout projects it). `kernel_source ==
    /// "vulkan-slang"` is the RED discriminator. Precision is contract-sourced and
    /// NOT asserted here.
    #[test]
    fn register_vulkan_kernels_binds_movement_family_from_contract() {
        let mut table = KernelBindingTable::new();
        register_vulkan_kernels(&mut table);
        let vk = BackendId::Vulkan;

        let f32 = DType::F32; let f16 = DType::F16; let bf16 = DType::BF16; let f64 = DType::F64;

        // Every movement binding is stride-aware (strided_input == true).
        let cases: &[(OpKind, &[DType], KernelRef)] = &[
            (OpKind::Concat, &[f32,  f32],  concat::concat_f32 as KernelRef),
            (OpKind::Concat, &[f16,  f16],  concat::concat_f16 as KernelRef),
            (OpKind::Concat, &[bf16, bf16], concat::concat_bf16 as KernelRef),
            (OpKind::Concat, &[f64,  f64],  concat::concat_f64 as KernelRef),
            (OpKind::CumSum, &[f32,  f32],  cumsum::cumsum_f32 as KernelRef),
            (OpKind::CumSum, &[f64,  f64],  cumsum::cumsum_f64 as KernelRef),
            (OpKind::CumSum, &[f16,  f16],  cumsum::cumsum_f16 as KernelRef),
            (OpKind::CumSum, &[bf16, bf16], cumsum::cumsum_bf16 as KernelRef),
        ];

        let mut checked = 0usize;
        for (op, key, expected) in cases {
            let alts = table.lookup_alternatives(*op, key, vk);
            let entry = alts.iter().find(|e| e.kernel as usize == *expected as usize);
            let entry = match entry {
                Some(e) => e,
                None => panic!(
                    "{op:?} {key:?}/Vulkan: production wrapper must be bound FROM the vulkan \
                     movement contract in register_vulkan_kernels; found {} alt(s) with sources {:?}",
                    alts.len(),
                    alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                ),
            };
            assert_eq!(
                entry.kernel_source, "vulkan-slang",
                "{op:?} {key:?}: movement family must be contract-sourced \
                 (kernel_source=\"vulkan-slang\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                entry.caps.strided_input,
                "{op:?} {key:?}: caps must ride through the import truthfully (strided_input=true)",
            );
            checked += 1;
        }
        assert_eq!(checked, 8, "all 8 (Concat/CumSum) movement keys checked");
    }
}

// ===========================================================================
// FKC contract-migration tests (born-red gate for the Vulkan shape family)
// ===========================================================================

#[cfg(test)]
mod shape_contract_tests {
    use super::*;
    use crate::kernel::{KernelBindingTable, KernelRef};

    /// EIGHTH VULKAN-BACKEND FKC CONSUMER (born-red gate). `register_vulkan_kernels`
    /// registers the whole Vulkan **shape** family — Triu (7 dtypes) + Tril (7) +
    /// Flip (7) + Roll (7), key `[T, T]` — FROM ITS FKC KERNEL CONTRACT
    /// (`docs/kernel-contracts/vulkan/shape.fkc.md`) via
    /// `register_vulkan_shape_from_contract`, the sole path now that the
    /// hand-written `register_with_precision(OpKind::{Triu,Tril}, …)` +
    /// `register_with_caps_and_precision(OpKind::{Flip,Roll}, …, strided, …)` regs
    /// are DELETED.
    ///
    /// Each op is ONE dtype-agnostic wrapper across its 7 dtype keys (a
    /// synthetic-base umbrella; the fan resolves `<base>_<suffix>` to the one
    /// wrapper). `kernel_source == "vulkan-slang"` (RED discriminator).
    /// `caps.strided_input` MATCHES per op: Triu/Tril contiguous (false),
    /// Flip/Roll stride-aware (true) — the caps-through-import proof. Precision is
    /// contract-sourced and NOT asserted here.
    #[test]
    fn register_vulkan_kernels_binds_shape_family_from_contract() {
        let mut table = KernelBindingTable::new();
        register_vulkan_kernels(&mut table);
        let vk = BackendId::Vulkan;

        let dts = [DType::F32, DType::F16, DType::BF16, DType::F64,
                   DType::I32, DType::U32, DType::I64];
        // (OpKind, wrapper, expected strided_input)
        let ops: &[(OpKind, KernelRef, bool)] = &[
            (OpKind::Triu, triangular::triu as KernelRef, false),
            (OpKind::Tril, triangular::tril as KernelRef, false),
            (OpKind::Flip, flip::flip as KernelRef, true),
            (OpKind::Roll, roll::roll as KernelRef, true),
        ];

        let mut checked = 0usize;
        for &dt in &dts {
            for (op, expected, expect_strided) in ops {
                let key = [dt, dt];
                let alts = table.lookup_alternatives(*op, &key, vk);
                let entry = alts.iter().find(|e| e.kernel as usize == *expected as usize);
                let entry = match entry {
                    Some(e) => e,
                    None => panic!(
                        "{op:?} {key:?}/Vulkan: production wrapper must be bound FROM the vulkan \
                         shape contract in register_vulkan_kernels; found {} alt(s) with sources {:?}",
                        alts.len(),
                        alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                    ),
                };
                assert_eq!(
                    entry.kernel_source, "vulkan-slang",
                    "{op:?} {key:?}: shape family must be contract-sourced \
                     (kernel_source=\"vulkan-slang\"); got {:?}",
                    entry.kernel_source,
                );
                assert_eq!(
                    entry.caps.strided_input, *expect_strided,
                    "{op:?} {key:?}: caps must ride through the import truthfully (strided_input)",
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 28, "all 28 (Triu/Tril/Flip/Roll × 7 dtypes) shape keys checked");
    }
}

// ===========================================================================
// FKC contract-migration tests (born-red gate for the Vulkan pad-copy family)
// ===========================================================================

#[cfg(test)]
mod pad_copy_contract_tests {
    use super::*;
    use crate::kernel::{KernelBindingTable, KernelRef};

    /// NINTH VULKAN-BACKEND FKC CONSUMER (born-red gate). `register_vulkan_kernels`
    /// registers the whole Vulkan **pad-copy** family — Pad (6 dtypes) +
    /// PadBackward (6) + Copy (9) — FROM ITS FKC KERNEL CONTRACT
    /// (`docs/kernel-contracts/vulkan/pad-copy.fkc.md`) via
    /// `register_vulkan_pad_copy_from_contract`, the sole path now that the
    /// hand-written `register_with_precision(OpKind::{Pad,PadBackward,Copy}, …)`
    /// regs are DELETED. Each op is ONE dtype-agnostic wrapper across its dtype keys
    /// (a synthetic-base umbrella). `kernel_source == "vulkan-slang"` (RED
    /// discriminator); `caps.strided_input == false` (contiguous-only byte movers).
    /// Precision is contract-sourced (byte-exact) and NOT asserted here.
    #[test]
    fn register_vulkan_kernels_binds_pad_copy_family_from_contract() {
        let mut table = KernelBindingTable::new();
        register_vulkan_kernels(&mut table);
        let vk = BackendId::Vulkan;

        let pad_dts = [DType::F32, DType::F16, DType::BF16, DType::F64, DType::U8, DType::U32];
        let copy_dts = [DType::F32, DType::F16, DType::BF16, DType::F64, DType::U32,
                        DType::U8, DType::I16, DType::I32, DType::I64];

        let mut checked = 0usize;
        let mut check = |op: OpKind, key: &[DType], expected: KernelRef| {
            let alts = table.lookup_alternatives(op, key, vk);
            let entry = alts.iter().find(|e| e.kernel as usize == expected as usize);
            let entry = match entry {
                Some(e) => e,
                None => panic!(
                    "{op:?} {key:?}/Vulkan: production wrapper must be bound FROM the vulkan \
                     pad-copy contract in register_vulkan_kernels; found {} alt(s) with sources {:?}",
                    alts.len(),
                    alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                ),
            };
            assert_eq!(
                entry.kernel_source, "vulkan-slang",
                "{op:?} {key:?}: pad-copy family must be contract-sourced \
                 (kernel_source=\"vulkan-slang\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "{op:?} {key:?}: caps preserved (contiguous-only, strided_input=false)",
            );
            checked += 1;
        };

        for &dt in &pad_dts {
            check(OpKind::Pad, &[dt, dt], pad::pad_const as KernelRef);
            check(OpKind::PadBackward, &[dt, dt], pad::pad_backward as KernelRef);
        }
        for &dt in &copy_dts {
            check(OpKind::Copy, &[dt, dt], copy_to_cpu_vulkan as KernelRef);
        }
        assert_eq!(checked, 21, "all 21 (Pad/PadBackward/Copy) pad-copy keys checked");
    }
}

// ===========================================================================
// FKC contract-migration tests (born-red gate for the Vulkan write-slice family)
// ===========================================================================

#[cfg(test)]
mod write_slice_contract_tests {
    use super::*;
    use crate::kernel::{KernelBindingTable, KernelRef};

    /// TENTH VULKAN-BACKEND FKC CONSUMER (born-red gate). `register_vulkan_kernels`
    /// registers the whole Vulkan **write-slice** family — WriteSlice (9 dtypes) +
    /// WriteSliceRotating (9), key `[T, T]` — FROM ITS FKC KERNEL CONTRACT
    /// (`docs/kernel-contracts/vulkan/write-slice.fkc.md`) via
    /// `register_vulkan_write_slice_from_contract`, the sole path now that the
    /// hand-written `register_with_precision(OpKind::{WriteSlice,WriteSliceRotating},
    /// …)` regs are DELETED. Each op is BYTE-WIDTH-keyed: 9 dtype keys resolve to 4
    /// byte-width wrappers (b1/b2/b4/b8) — the cast family's "several sections share
    /// one wrapper" precedent. `kernel_source == "vulkan-slang"` (RED discriminator);
    /// `caps.strided_input == false` (contiguous slab writes). Precision is
    /// contract-sourced (byte-exact) and NOT asserted here.
    #[test]
    fn register_vulkan_kernels_binds_write_slice_family_from_contract() {
        let mut table = KernelBindingTable::new();
        register_vulkan_kernels(&mut table);
        let vk = BackendId::Vulkan;

        // (dtype, WriteSlice wrapper, WriteSliceRotating wrapper) by byte width.
        let cases: &[(DType, KernelRef, KernelRef)] = &[
            (DType::F32, write_slice::write_slice_b4 as KernelRef, write_slice_rotating::write_slice_rotating_b4 as KernelRef),
            (DType::I32, write_slice::write_slice_b4 as KernelRef, write_slice_rotating::write_slice_rotating_b4 as KernelRef),
            (DType::U32, write_slice::write_slice_b4 as KernelRef, write_slice_rotating::write_slice_rotating_b4 as KernelRef),
            (DType::F16, write_slice::write_slice_b2 as KernelRef, write_slice_rotating::write_slice_rotating_b2 as KernelRef),
            (DType::BF16, write_slice::write_slice_b2 as KernelRef, write_slice_rotating::write_slice_rotating_b2 as KernelRef),
            (DType::F64, write_slice::write_slice_b8 as KernelRef, write_slice_rotating::write_slice_rotating_b8 as KernelRef),
            (DType::I64, write_slice::write_slice_b8 as KernelRef, write_slice_rotating::write_slice_rotating_b8 as KernelRef),
            (DType::U8, write_slice::write_slice_b1 as KernelRef, write_slice_rotating::write_slice_rotating_b1 as KernelRef),
            (DType::I8, write_slice::write_slice_b1 as KernelRef, write_slice_rotating::write_slice_rotating_b1 as KernelRef),
        ];

        let mut checked = 0usize;
        let mut check = |op: OpKind, key: &[DType], expected: KernelRef| {
            let alts = table.lookup_alternatives(op, key, vk);
            let entry = alts.iter().find(|e| e.kernel as usize == expected as usize);
            let entry = match entry {
                Some(e) => e,
                None => panic!(
                    "{op:?} {key:?}/Vulkan: production wrapper must be bound FROM the vulkan \
                     write-slice contract in register_vulkan_kernels; found {} alt(s) with sources {:?}",
                    alts.len(),
                    alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                ),
            };
            assert_eq!(
                entry.kernel_source, "vulkan-slang",
                "{op:?} {key:?}: write-slice family must be contract-sourced \
                 (kernel_source=\"vulkan-slang\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "{op:?} {key:?}: caps preserved (contiguous-only, strided_input=false)",
            );
            checked += 1;
        };

        for (dt, ws, wsr) in cases {
            check(OpKind::WriteSlice, &[*dt, *dt], *ws);
            check(OpKind::WriteSliceRotating, &[*dt, *dt], *wsr);
        }
        assert_eq!(checked, 18, "all 18 (WriteSlice/WriteSliceRotating × 9 dtypes) write-slice keys checked");
    }
}

// ===========================================================================
// FKC contract-migration tests (born-red gate for the Vulkan rope family)
// ===========================================================================

#[cfg(test)]
mod rope_contract_tests {
    use super::*;
    use crate::kernel::{KernelBindingTable, KernelRef};

    /// ELEVENTH VULKAN-BACKEND FKC CONSUMER (born-red gate). `register_vulkan_kernels`
    /// registers the whole Vulkan **rope** family — RoPE (4 per-dtype), 4-slot key
    /// `[x, cos, sin, out]` — FROM ITS FKC KERNEL CONTRACT
    /// (`docs/kernel-contracts/vulkan/rope.fkc.md`) via
    /// `register_vulkan_rope_from_contract`, the sole path now that the hand-written
    /// `register_with_caps_and_precision(OpKind::Rope, …, strided, …)` regs are
    /// DELETED. One section fans `x` + `cos` + `sin` over the SAME `[F32,F16,F64,BF16]`
    /// list (§3.4), keying `[T, T, T, T]`.
    ///
    /// RoPE is STRIDE-AWARE (the deleted regs carried `strided`), so
    /// `caps.strided_input == true` is the caps-through-import proof here.
    /// `kernel_source == "vulkan-slang"` is the RED discriminator. Precision is
    /// contract-sourced and NOT asserted here.
    #[test]
    fn register_vulkan_kernels_binds_rope_family_from_contract() {
        let mut table = KernelBindingTable::new();
        register_vulkan_kernels(&mut table);
        let vk = BackendId::Vulkan;

        let f32 = DType::F32; let f16 = DType::F16; let bf16 = DType::BF16; let f64 = DType::F64;
        let cases: &[(&[DType], KernelRef)] = &[
            (&[f32,  f32,  f32,  f32],  attention::rope_f32 as KernelRef),
            (&[f16,  f16,  f16,  f16],  attention::rope_f16 as KernelRef),
            (&[f64,  f64,  f64,  f64],  attention::rope_f64 as KernelRef),
            (&[bf16, bf16, bf16, bf16], attention::rope_bf16 as KernelRef),
        ];

        let mut checked = 0usize;
        for (key, expected) in cases {
            let alts = table.lookup_alternatives(OpKind::Rope, key, vk);
            let entry = alts.iter().find(|e| e.kernel as usize == *expected as usize);
            let entry = match entry {
                Some(e) => e,
                None => panic!(
                    "Rope {key:?}/Vulkan: production wrapper must be bound FROM the vulkan \
                     rope contract in register_vulkan_kernels; found {} alt(s) with sources {:?}",
                    alts.len(),
                    alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                ),
            };
            assert_eq!(
                entry.kernel_source, "vulkan-slang",
                "Rope {key:?}: rope family must be contract-sourced \
                 (kernel_source=\"vulkan-slang\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                entry.caps.strided_input,
                "Rope {key:?}: caps must ride through the import truthfully (strided_input=true)",
            );
            checked += 1;
        }
        assert_eq!(checked, 4, "all 4 (RoPE × dtype) rope keys checked");
    }
}

// ===========================================================================
// FKC contract-migration tests (born-red gate for the Vulkan reduce family)
// ===========================================================================

#[cfg(test)]
mod reduce_contract_tests {
    use super::*;
    use crate::kernel::{KernelBindingTable, KernelRef};

    /// TWELFTH VULKAN-BACKEND FKC CONSUMER (born-red gate). `register_vulkan_kernels`
    /// registers the whole Vulkan **reduce** family — SumReduce / MaxReduce /
    /// MinReduce / MeanReduce (4 per-dtype each = 16, key `[T, T]`) + ArgMaxDim /
    /// ArgMinDim (4 per-dtype each = 8, key `[T, U32]`) — FROM ITS FKC KERNEL
    /// CONTRACT (`docs/kernel-contracts/vulkan/reduce.fkc.md`) via
    /// `register_vulkan_reduce_from_contract`, the sole path now that the
    /// hand-written `register_with_precision(OpKind::{SumReduce,…,ArgMinDim}, …)`
    /// regs are DELETED. Each OpKind is its own section fanned over the dtype list to
    /// its distinct per-dtype wrapper. `kernel_source == "vulkan-slang"` (RED
    /// discriminator); `caps.strided_input == false` (contiguous subgroup-tree
    /// reductions). Precision is contract-sourced and NOT asserted here.
    #[test]
    fn register_vulkan_kernels_binds_reduce_family_from_contract() {
        let mut table = KernelBindingTable::new();
        register_vulkan_kernels(&mut table);
        let vk = BackendId::Vulkan;
        let u32 = DType::U32;

        let mut checked = 0usize;
        let mut check = |op: OpKind, key: &[DType], expected: KernelRef| {
            let alts = table.lookup_alternatives(op, key, vk);
            let entry = alts.iter().find(|e| e.kernel as usize == expected as usize);
            let entry = match entry {
                Some(e) => e,
                None => panic!(
                    "{op:?} {key:?}/Vulkan: production wrapper must be bound FROM the vulkan \
                     reduce contract in register_vulkan_kernels; found {} alt(s) with sources {:?}",
                    alts.len(),
                    alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                ),
            };
            assert_eq!(
                entry.kernel_source, "vulkan-slang",
                "{op:?} {key:?}: reduce family must be contract-sourced \
                 (kernel_source=\"vulkan-slang\"); got {:?}",
                entry.kernel_source,
            );
            assert!(!entry.caps.strided_input, "{op:?} {key:?}: contiguous-only (strided_input=false)");
            checked += 1;
        };

        // Value reduces: 4 ops × 4 dtypes, key [T, T].
        let val: &[(OpKind, [KernelRef; 4])] = &[
            (OpKind::SumReduce,  [reduce::sum_f32, reduce::sum_f16, reduce::sum_bf16, reduce::sum_f64]),
            (OpKind::MaxReduce,  [reduce::max_f32, reduce::max_f16, reduce::max_bf16, reduce::max_f64]),
            (OpKind::MinReduce,  [reduce::min_f32, reduce::min_f16, reduce::min_bf16, reduce::min_f64]),
            (OpKind::MeanReduce, [reduce::mean_f32, reduce::mean_f16, reduce::mean_bf16, reduce::mean_f64]),
        ];
        let dts = [DType::F32, DType::F16, DType::BF16, DType::F64];
        for (op, wraps) in val {
            for (i, &dt) in dts.iter().enumerate() {
                check(*op, &[dt, dt], wraps[i]);
            }
        }
        // Arg reduces: 2 ops × 4 dtypes, key [T, U32].
        let arg: &[(OpKind, [KernelRef; 4])] = &[
            (OpKind::ArgMaxDim, [arg_reduce::argmax_f32, arg_reduce::argmax_f16, arg_reduce::argmax_bf16, arg_reduce::argmax_f64]),
            (OpKind::ArgMinDim, [arg_reduce::argmin_f32, arg_reduce::argmin_f16, arg_reduce::argmin_bf16, arg_reduce::argmin_f64]),
        ];
        for (op, wraps) in arg {
            for (i, &dt) in dts.iter().enumerate() {
                check(*op, &[dt, u32], wraps[i]);
            }
        }
        assert_eq!(checked, 24, "all 24 (Sum/Max/Min/Mean + ArgMax/ArgMin × 4 dtypes) reduce keys checked");
    }
}

// ===========================================================================
// FKC contract-migration tests (born-red gate for the Vulkan norm family)
// ===========================================================================

#[cfg(test)]
mod norm_contract_tests {
    use super::*;
    use crate::kernel::{KernelBindingTable, KernelRef};

    /// THIRTEENTH VULKAN-BACKEND FKC CONSUMER (born-red gate). `register_vulkan_kernels`
    /// registers the whole Vulkan **norm** family — SoftmaxLastDim (4, key `[T, T]`),
    /// SoftmaxLastDimBackward (4, key `[y, g, dx]`), LayerNormLastDim (4, `[T, T]`),
    /// LayerNormLastDimBackward (4, `[x, g, dx]`), RmsNormLastDim (4, `[T, T]`) —
    /// FROM ITS FKC KERNEL CONTRACT (`docs/kernel-contracts/vulkan/norm.fkc.md`) via
    /// `register_vulkan_norm_from_contract`, the sole path now that the hand-written
    /// `register_with_precision(OpKind::{SoftmaxLastDim,…,RmsNormLastDim}, …)` regs
    /// are DELETED. Each OpKind is its own section fanned over `[F32, F16, BF16, F64]`
    /// to its distinct per-dtype wrapper. `kernel_source == "vulkan-slang"` (RED
    /// discriminator); `caps.strided_input == false` (contiguous per-row
    /// reductions). Precision is contract-sourced and NOT asserted here.
    #[test]
    fn register_vulkan_kernels_binds_norm_family_from_contract() {
        let mut table = KernelBindingTable::new();
        register_vulkan_kernels(&mut table);
        let vk = BackendId::Vulkan;

        let mut checked = 0usize;
        let mut check = |op: OpKind, key: &[DType], expected: KernelRef| {
            let alts = table.lookup_alternatives(op, key, vk);
            let entry = alts.iter().find(|e| e.kernel as usize == expected as usize);
            let entry = match entry {
                Some(e) => e,
                None => panic!(
                    "{op:?} {key:?}/Vulkan: production wrapper must be bound FROM the vulkan \
                     norm contract in register_vulkan_kernels; found {} alt(s) with sources {:?}",
                    alts.len(),
                    alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                ),
            };
            assert_eq!(
                entry.kernel_source, "vulkan-slang",
                "{op:?} {key:?}: norm family must be contract-sourced \
                 (kernel_source=\"vulkan-slang\"); got {:?}",
                entry.kernel_source,
            );
            assert!(!entry.caps.strided_input, "{op:?} {key:?}: contiguous-only (strided_input=false)");
            checked += 1;
        };

        let dts = [DType::F32, DType::F16, DType::BF16, DType::F64];
        // Unary [T, T] families.
        let un: &[(OpKind, [KernelRef; 4])] = &[
            (OpKind::SoftmaxLastDim,  [softmax::softmax_f32, softmax::softmax_f16, softmax::softmax_bf16, softmax::softmax_f64]),
            (OpKind::LayerNormLastDim,[norm::layer_norm_f32, norm::layer_norm_f16, norm::layer_norm_bf16, norm::layer_norm_f64]),
            (OpKind::RmsNormLastDim,  [norm::rms_f32, norm::rms_f16, norm::rms_bf16, norm::rms_f64]),
        ];
        for (op, wraps) in un {
            for (i, &dt) in dts.iter().enumerate() { check(*op, &[dt, dt], wraps[i]); }
        }
        // Backward [T, T, T] families.
        let bwd: &[(OpKind, [KernelRef; 4])] = &[
            (OpKind::SoftmaxLastDimBackward, [softmax::softmax_last_dim_backward_f32, softmax::softmax_last_dim_backward_f16, softmax::softmax_last_dim_backward_bf16, softmax::softmax_last_dim_backward_f64]),
            (OpKind::LayerNormLastDimBackward, [norm::layer_norm_backward_f32, norm::layer_norm_backward_f16, norm::layer_norm_backward_bf16, norm::layer_norm_backward_f64]),
        ];
        for (op, wraps) in bwd {
            for (i, &dt) in dts.iter().enumerate() { check(*op, &[dt, dt, dt], wraps[i]); }
        }
        assert_eq!(checked, 20, "all 20 (Softmax/SoftmaxBwd/LayerNorm/LayerNormBwd/RmsNorm × 4 dtypes) norm keys checked");
    }
}

// ===========================================================================
// FKC contract-migration tests (born-red gate for the Vulkan QMatMul family)
// ===========================================================================
//
// Same `#![cfg(feature = "vulkan")]` file gate as the cast/…/norm tests above:
// this module compiles only under `--features vulkan` and touches NO device
// (registration is pure binding-table population).

#[cfg(test)]
mod qmatmul_contract_tests {
    use super::*;
    use crate::kernel::{KernelBindingTable, KernelRef};

    /// FOURTEENTH VULKAN-BACKEND FKC CONSUMER (born-red gate). `register_vulkan_kernels`
    /// registers the single Vulkan **QMatMul** binding (`(QMatMul, [F32, U32, F32],
    /// Vulkan)` — F32 activation × U32-typed GGML block-quant weight → F32 output, the
    /// one wrapper `qmatmul::qmatmul_vk` route-picking Q4_0 / Q4_K_M / Q8_0 internally
    /// by `OpParams::QMatMul.quant_type`) FROM ITS RE-AUTHORED FKC KERNEL CONTRACT
    /// (`docs/kernel-contracts/vulkan/qmatmul.fkc.md`) via
    /// `register_vulkan_qmatmul_from_contract` — the sole registration path now that
    /// the hand-written `table.register_with_precision(OpKind::QMatMul, …)` reg is
    /// DELETED.
    ///
    /// The corpus's `vulkan/quantized.fkc.md` models QMATMUL as TWO aspirational
    /// `fused_op: QMATMUL` Q4_0 sections (`qmatvec_q4_0` gemv + `matmul_q4_0_tiled`),
    /// which mismatch production on BOTH axes: production registers the PRIMITIVE
    /// `OpKind::QMatMul` (not the fused op) keyed with the LOGICAL U32 weight dtype
    /// (not the physical U8 block stream). So the migration authors ONE faithful
    /// single-dtype-per-operand `op_kind: QMatMul` section in the new production
    /// `docs/kernel-contracts/vulkan/qmatmul.fkc.md` (weight `U32`, `fdx.quant`
    /// GGML_BLOCK keeping byte-honesty — the CPU linear-quant U8→U32 precedent), and
    /// flips the two corpus fused sections to `registrable: false` describe-only
    /// cross-referencing it (the matmul_mixed_precision precedent).
    ///
    /// This asserts: the `(QMatMul, [F32, U32, F32], Vulkan)` key resolves to the
    /// EXACT production wrapper fn-pointer `qmatmul::qmatmul_vk` (behavior-preserving);
    /// `kernel_source == "vulkan-slang"` (the RED discriminator — the deleted hand
    /// path stamped `""`, so before the import is wired this assert fails on the empty
    /// source); and `caps.strided_input == false` — the contiguous-only truth of the
    /// deleted `register_with_precision` reg (the GGML block stream has a fixed block
    /// layout; a strided operand is auto-Contiguized first). Precision is
    /// contract-sourced (the matmul/conv precedent: the hand-written
    /// `VULKAN_QMATMUL_PRECISION` over-claimed `bit_stable_on_same_hardware: true`; the
    /// contract seeds the truthful audited-nondeterministic posture) and NOT asserted
    /// here.
    #[test]
    fn register_vulkan_kernels_binds_qmatmul_from_contract() {
        let mut table = KernelBindingTable::new();
        register_vulkan_kernels(&mut table);
        let vk = BackendId::Vulkan;

        let key: &[DType] = &[DType::F32, DType::U32, DType::F32];
        let expected = qmatmul::qmatmul_vk as KernelRef;
        let alts = table.lookup_alternatives(OpKind::QMatMul, key, vk);
        let entry = alts.iter().find(|e| e.kernel as usize == expected as usize);
        let entry = match entry {
            Some(e) => e,
            None => panic!(
                "QMatMul {key:?}/Vulkan: production wrapper (qmatmul_vk) must be bound FROM the \
                 vulkan qmatmul contract in register_vulkan_kernels; found {} alt(s) with sources {:?}",
                alts.len(),
                alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
            ),
        };
        assert_eq!(
            entry.kernel_source, "vulkan-slang",
            "QMatMul {key:?}: qmatmul must be contract-sourced \
             (kernel_source=\"vulkan-slang\"); got {:?}",
            entry.kernel_source,
        );
        assert!(
            !entry.caps.strided_input,
            "QMatMul {key:?}: caps preserved (contiguous-only, strided_input=false)",
        );
    }
}

// ===========================================================================
// FKC contract-migration tests (born-red gate for the Vulkan FlashAttn family)
// ===========================================================================
//
// Same `#![cfg(feature = "vulkan")]` file gate as the cast/…/qmatmul tests
// above: this module compiles only under `--features vulkan` and touches NO
// device (registration is pure binding-table population).

#[cfg(test)]
mod attention_contract_tests {
    use super::*;
    use crate::kernel::{KernelBindingTable, KernelRef};

    /// FIFTEENTH (and LAST) VULKAN-BACKEND FKC CONSUMER (born-red gate).
    /// `register_vulkan_kernels` registers the whole Vulkan **FlashAttention**
    /// family (12 `KernelRef`s: forward `OpKind::FlashAttn` at f32/bf16/f16 +
    /// backward `OpKind::FlashAttnBackward{Q,K,V}` at f32, each with a
    /// no-alibi AND a with-alibi binding shape) FROM ITS AUTHORED FKC KERNEL
    /// CONTRACT (`docs/kernel-contracts/vulkan/attention.fkc.md`) via
    /// `register_vulkan_attention_from_contract` — the sole registration path
    /// now that the hand-written `table.register_with_precision(OpKind::{FlashAttn,
    /// FlashAttnBackwardQ,FlashAttnBackwardK,FlashAttnBackwardV}, …)` regs are
    /// DELETED. This was the LAST hand-written `table.register*` call in
    /// `register_vulkan_kernels`, so the Vulkan backend is now 100%
    /// contract-sourced.
    ///
    /// The forward wrappers are DISTINCT per dtype
    /// (`flash_attn::flash_attn_{f32,bf16,f16}`), so the contract authors THREE
    /// single-dtype-per-operand forward sections (the conv2d + CPU-attention
    /// per-dtype-section precedent) rather than one dtype-fanned umbrella; the
    /// three backward selectors are f32-only single sections. In every section
    /// `alibi_slopes` is an `optional: true` LAST input, so the importer's
    /// optional-last fan (§3.4) registers EACH as TWO keys (with + without the
    /// operand) — byte-for-byte the deleted regs' dual `[q,k,v,out]` /
    /// `[q,k,v,alibi,out]` (forward) and `[q,k,v,do,out]` / `[q,k,v,do,alibi,out]`
    /// (backward) shapes.
    ///
    /// For each `(OpKind, [T..], Vulkan)` key this asserts: the binding resolves
    /// to the EXACT production wrapper fn-pointer (behavior-preserving);
    /// `kernel_source == "vulkan-slang"` (the RED discriminator — the deleted
    /// hand path stamped `""`, so before the import is wired this assert fails on
    /// the empty source); and `caps.strided_input == false` — the contiguous-only
    /// truth of the deleted plain `register_with_precision` regs. Precision is
    /// contract-sourced (the retired `VULKAN_{FLOAT,HALF}_POINTWISE_PRECISION`
    /// consts over-claimed `bit_stable_on_same_hardware: true` for a shared-mem
    /// softmax reduction; the contract seeds the truthful audited-nondeterministic
    /// posture, matching the aspirational `conv-attn-rope.fkc.md`) and NOT
    /// asserted here.
    #[test]
    fn register_vulkan_kernels_binds_flash_attn_family_from_contract() {
        let mut table = KernelBindingTable::new();
        register_vulkan_kernels(&mut table);
        let vk = BackendId::Vulkan;

        // (OpKind, key dtypes [inputs.., out], expected production wrapper). Both the
        // no-alibi and the with-alibi (one extra pre-output slot) shapes per op/dtype.
        // Every FlashAttn binding is contiguous-only (strided_input == false), checked
        // uniformly below.
        let cases: &[(OpKind, &[DType], KernelRef)] = &[
            // ---- Forward FlashAttn: f32 / bf16 / f16, no-alibi [q,k,v,out] + with-alibi [q,k,v,alibi,out]. ----
            (OpKind::FlashAttn, &[DType::F32, DType::F32, DType::F32, DType::F32], flash_attn::flash_attn_f32 as KernelRef),
            (OpKind::FlashAttn, &[DType::F32, DType::F32, DType::F32, DType::F32, DType::F32], flash_attn::flash_attn_f32 as KernelRef),
            (OpKind::FlashAttn, &[DType::BF16, DType::BF16, DType::BF16, DType::BF16], flash_attn::flash_attn_bf16 as KernelRef),
            (OpKind::FlashAttn, &[DType::BF16, DType::BF16, DType::BF16, DType::BF16, DType::BF16], flash_attn::flash_attn_bf16 as KernelRef),
            (OpKind::FlashAttn, &[DType::F16, DType::F16, DType::F16, DType::F16], flash_attn::flash_attn_f16 as KernelRef),
            (OpKind::FlashAttn, &[DType::F16, DType::F16, DType::F16, DType::F16, DType::F16], flash_attn::flash_attn_f16 as KernelRef),
            // ---- Backward Q/K/V f32: no-alibi [q,k,v,do,out] + with-alibi [q,k,v,do,alibi,out]. ----
            (OpKind::FlashAttnBackwardQ, &[DType::F32, DType::F32, DType::F32, DType::F32, DType::F32], flash_attn::flash_attn_backward_q_f32 as KernelRef),
            (OpKind::FlashAttnBackwardQ, &[DType::F32, DType::F32, DType::F32, DType::F32, DType::F32, DType::F32], flash_attn::flash_attn_backward_q_f32 as KernelRef),
            (OpKind::FlashAttnBackwardK, &[DType::F32, DType::F32, DType::F32, DType::F32, DType::F32], flash_attn::flash_attn_backward_k_f32 as KernelRef),
            (OpKind::FlashAttnBackwardK, &[DType::F32, DType::F32, DType::F32, DType::F32, DType::F32, DType::F32], flash_attn::flash_attn_backward_k_f32 as KernelRef),
            (OpKind::FlashAttnBackwardV, &[DType::F32, DType::F32, DType::F32, DType::F32, DType::F32], flash_attn::flash_attn_backward_v_f32 as KernelRef),
            (OpKind::FlashAttnBackwardV, &[DType::F32, DType::F32, DType::F32, DType::F32, DType::F32, DType::F32], flash_attn::flash_attn_backward_v_f32 as KernelRef),
        ];

        let mut checked = 0usize;
        for (op, key, expected) in cases {
            let alts = table.lookup_alternatives(*op, key, vk);
            let entry = alts.iter().find(|e| e.kernel as usize == *expected as usize);
            let entry = match entry {
                Some(e) => e,
                None => panic!(
                    "{op:?} {key:?}/Vulkan: production wrapper must be bound FROM the vulkan \
                     attention contract in register_vulkan_kernels; found {} alt(s) with sources {:?}",
                    alts.len(),
                    alts.iter().map(|e| e.kernel_source).collect::<Vec<_>>(),
                ),
            };
            assert_eq!(
                entry.kernel_source, "vulkan-slang",
                "{op:?} {key:?}: FlashAttn family must be contract-sourced \
                 (kernel_source=\"vulkan-slang\"); got {:?}",
                entry.kernel_source,
            );
            assert!(
                !entry.caps.strided_input,
                "{op:?} {key:?}: caps preserved (contiguous-only, strided_input=false)",
            );
            checked += 1;
        }
        assert_eq!(checked, 12, "all 12 FlashAttn family bindings (6 forward + 6 backward) checked");
    }
}
