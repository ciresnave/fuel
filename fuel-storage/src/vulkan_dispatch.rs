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
        let (outer_count, last_dim) = match params {
            OpParams::SoftmaxLastDim { outer_count, last_dim } => {
                (*outer_count, *last_dim)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::softmax::softmax_f32: expected OpParams::SoftmaxLastDim, got {:?}",
                    other,
                )).bt());
            }
        };
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
}

// ===========================================================================
// Norm — RmsNorm last-dim f32 (V.2.C)
// ===========================================================================
//
// LayerNorm is V.3 (no Slang kernel yet). RmsNorm pulls
// `(outer_count, last_dim, eps)` from `OpParams::NormLastDim`.

pub mod norm {
    use super::*;

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
        let (outer_count, last_dim, eps) = match params {
            OpParams::NormLastDim { outer_count, last_dim, eps } => {
                (*outer_count, *last_dim, *eps)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::norm::rms_f32: expected OpParams::NormLastDim, got {:?}",
                    other,
                )).bt());
            }
        };
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
// Cast — f32↔f16, f32↔bf16 (V.3.B)
// ===========================================================================
//
// One wrapper handles all 4 supported (src, dst) pairs by inspecting
// Storage.dtype. Element count derived from src bytes / src elem-size.
// Requires `n` even (half-precision dtypes are u32-packed 2-per-word);
// odd-count tensors return Err and the route picker falls back to CPU.

pub mod cast {
    use super::*;

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

pub mod concat {
    use super::*;
    use crate::BackendStorage;
    use fuel_core_types::Shape;

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
        let (outer_count, input_dim_sizes, inner_count) = match params {
            OpParams::Concat { outer_count, input_dim_sizes, inner_count } => {
                (*outer_count, input_dim_sizes.clone(), *inner_count)
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
        // Recover the concat dim: outer_count = prod(dims[..dim]).
        let a_dims = layouts[0].shape().dims();
        let mut acc = 1usize;
        let mut dim_opt: Option<usize> = None;
        for (i, d) in a_dims.iter().enumerate() {
            if acc == outer_count {
                dim_opt = Some(i);
                break;
            }
            acc = acc.saturating_mul(*d);
        }
        let dim = dim_opt.or_else(|| {
            if outer_count == 1 { Some(0) } else { None }
        }).ok_or_else(|| {
            Error::Msg(format!(
                "vulkan_dispatch::concat::concat_f32: couldn't recover concat dim from \
                 outer_count={outer_count} + a_dims={a_dims:?}",
            )).bt()
        })?;

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
        // concat-dim size is `cum_dim`.
        let make_layout = |cum_dim: usize| -> Layout {
            let mut dims = a_dims.to_vec();
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

pub mod reduce {
    use super::*;

    vk_reduce_f32_wrapper!(sum_f32,  0, "reduce_sum_f32");
    vk_reduce_f32_wrapper!(max_f32,  1, "reduce_max_f32");
    vk_reduce_f32_wrapper!(min_f32,  2, "reduce_min_f32");
    // V.3.A.2: Mean added to reduce.slang + reduce_last_dim.slang as
    // op_id=3 (sum then divide by element count).
    vk_reduce_f32_wrapper!(mean_f32, 3, "reduce_mean_f32");
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
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::flip::flip: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (outer_count, dim_size, inner_count) = match params {
            OpParams::Flip { outer_count, dim_size, inner_count } => {
                (*outer_count, *dim_size, *inner_count)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::flip::flip: expected OpParams::Flip, got {:?}",
                    other,
                )).bt());
            }
        };
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
        backend.flip_bytes(byte_width, a, out, outer_count, dim_size, inner_count)
    }
}

pub mod roll {
    use super::*;

    pub fn roll(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        params: &OpParams,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(Error::Msg(format!(
                "vulkan_dispatch::roll::roll: expected 1 input + 1 output, got {} + {}",
                inputs.len(), outputs.len(),
            )).bt());
        }
        let (outer_count, dim_size, inner_count, shift) = match params {
            OpParams::Roll { outer_count, dim_size, inner_count, shift } => {
                (*outer_count, *dim_size, *inner_count, *shift)
            }
            other => {
                return Err(Error::Msg(format!(
                    "vulkan_dispatch::roll::roll: expected OpParams::Roll, got {:?}",
                    other,
                )).bt());
            }
        };
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
        backend.roll_bytes(byte_width, a, out, outer_count, dim_size, inner_count, shift)
    }
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
    let bytes = backend.download_bytes(vk_src)?;
    let mut out_guard = write_storage(&outputs[0])?;
    let dst = crate::dispatch::cpu_output(&mut out_guard)?;
    let n = bytes.len().min(dst.len_bytes());
    dst.bytes_mut()[..n].copy_from_slice(&bytes[..n]);
    Ok(())
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

    // ----- IndexSelect (V.2.D, f32 src + u32 ids) — pure gather (byte-level).
    // strided-input candidate: index_select.slang flattens input to
    // (outer, axis, inner) and reads via own-shape strides. Could be
    // extended to walk arbitrary input layout strides at gather time
    // (saves an auto-Contiguize when the source is a transpose view).
    // Indices are u32; their layout-strided variant is a follow-up. -----
    let idx_dts = [f32, DType::U32, f32];
    table.register_with_precision(OpKind::IndexSelect, &idx_dts, vk, indexing::index_select_f32, VULKAN_BYTE_LEVEL_PRECISION);

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

    // ----- Concat f32 (V.2.D) — memcpy (byte-level).
    // STRIDE-AWARE: concat_along_dim.slang explicitly supports per-
    // operand stride support so either input may be a lazy view
    // (permute, broadcast). N==2 only; N>2 falls back to next alt. -----
    table.register_with_caps_and_precision(OpKind::Concat, &u(f32), vk, concat::concat_f32, strided, VULKAN_BYTE_LEVEL_PRECISION);

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

    // ----- Affine f32 (V.2.E) — y = mul*x + add. Pointwise FMA.
    // STRIDE-AWARE (converted 2026-05-24): affine.slang now carries
    // the same per-dim shape/stride + flags Params as unary.slang.
    // `affine_f32_bytes` packs the layout identically to the unary
    // path. -----
    table.register_with_caps_and_precision(OpKind::Affine, &u(f32), vk, affine::affine_f32, strided, VULKAN_FLOAT_POINTWISE_PRECISION);

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

    // ----- Triu / Tril / Flip / Roll — pure byte-level (memcpy/mask).
    // CONTIGUOUS-ONLY by design: flip_b4 / triu_b4 / tril_b4 / roll_b4
    // view inputs as a flat (outer, dim_size, inner) 3-tuple and index
    // via own-shape strides. Arbitrary layout strides would require
    // a different decomposition; auto-Contiguize handles non-contiguous
    // inputs upstream. -----
    for &dt in &[DType::F32, DType::F16, DType::BF16, DType::F64,
                 DType::I32, DType::U32, DType::I64] {
        table.register_with_precision(OpKind::Triu, &u(dt), vk, triangular::triu, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::Tril, &u(dt), vk, triangular::tril, VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::Flip, &u(dt), vk, flip::flip,       VULKAN_BYTE_LEVEL_PRECISION);
        table.register_with_precision(OpKind::Roll, &u(dt), vk, roll::roll,       VULKAN_BYTE_LEVEL_PRECISION);
    }

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
