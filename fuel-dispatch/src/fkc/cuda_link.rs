//! The built-in CUDA (baracuda) backend's FKC `link_registry`
//! (kernel-seam-interop §3.5, §4.3; FKC §12.6). The CUDA analogue of
//! [`crate::fkc::CpuLinkRegistry`] / [`crate::fkc::VulkanLinkRegistry`]: maps
//! each CUDA kernel contract's `entry_point` symbol to the production dispatch
//! wrapper — the real, non-stub resolution the importer uses so an imported
//! contract binds the **actual** kernel (no raw pointers in the serialized
//! contract, FKC P9).
//!
//! For the built-in CUDA backend the wrappers live in
//! [`crate::baracuda_dispatch`] (the dispatch layer that adapts the baracuda
//! FFI kernels to [`KernelRef`]) and this table co-locates with them in
//! fuel-dispatch — exactly as [`crate::fkc::vulkan_link`] co-locates with
//! [`crate::vulkan_dispatch`]. An *external* provider instead exports its own
//! link registry across the FFI; this is Fuel's internal baracuda-provider
//! analogue.
//!
//! Everything here is gated behind the `cuda` cargo feature — the DEFAULT
//! build never compiles this module (mirroring `vulkan_link`).

use crate::fkc::lower::LinkRegistry;
use crate::kernel::{CostFn, KernelRef};

/// One `(contract entry_point symbol, production wrapper)` pair. The symbol
/// matches the contract's fanned `entry_point`: a section declares a BASE
/// `entry_point: "fuel_cuda_backend::fkc::<base>"` and the importer fans it to
/// `<base>_<dtype_suffix>` (§3.4), so this macro takes the bare fanned symbol
/// name + the fully-qualified wrapper path and rebuilds the full symbol — the
/// `fuel_vulkan_backend::fkc::` → `fuel_cuda_backend::fkc::` mirror of
/// [`crate::fkc::vulkan_link`]'s `vk_ep!`.
macro_rules! cuda_ep {
    ($sym:literal, $wrapper:path $(,)?) => {
        (
            concat!("fuel_cuda_backend::fkc::", $sym),
            $wrapper as KernelRef,
        )
    };
}

/// The CUDA cast (dtype-conversion) family's `symbol → production wrapper` map
/// — the FULL family: 70 `(SRC, DST)` pairs, EVERY one resolving to the SAME
/// dtype-agnostic wrapper [`crate::baracuda_dispatch::cast::cast_baracuda_wrapper`].
/// Contract: `docs/kernel-contracts/cuda/cast.fkc.md`, authored per-destination
/// (`## cast_to_<dst>`) with a `src`-operand dtype fan (§3.4). Unlike the CPU
/// cast family (per-target distinct wrappers) or the Vulkan cast family (three
/// structural wrappers), the CUDA cast is a SINGLE `cast_baracuda_wrapper` that
/// reads both dtypes off the in/out Storage and dispatches into baracuda's 8×8
/// FFI surface — so this is a **synthetic-base umbrella**: distinct
/// `(Cast, [SRC, DST])` keys are legal sibling registrations of one wrapper (the
/// Vulkan shape / pad-copy precedent).
///
/// The per-destination sections fan the `src` operand over its accepted sources,
/// so the importer resolves `cast_to_<dst>_<src_suffix>` — every symbol below is
/// exactly that fanned form. The 70 pairs are production's key set byte-for-byte:
/// the 8×8 cross product over `{F32, F64, F16, BF16, I32, U32, I64, U8}` (incl.
/// the `src == dst` diagonal), plus `F8E4M3 → {F32, F16, BF16}` and
/// `{F32, F16, BF16} → F8E4M3`. Caps ride through contiguous-only
/// (`requires_contiguous` ⇒ `strided_input == false`), byte-for-byte the deleted
/// hand-written `table.register(OpKind::Cast, …)` regs.
pub static CUDA_CAST_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // ----- cast_to_f32: 9 sources (8-set + F8E4M3). -----
    cuda_ep!("cast_to_f32_f32",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f32_f64",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f32_f16",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f32_bf16",   crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f32_i32",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f32_u32",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f32_i64",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f32_u8",     crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f32_f8e4m3", crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    // ----- cast_to_f16: 9 sources (8-set + F8E4M3). -----
    cuda_ep!("cast_to_f16_f32",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f16_f64",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f16_f16",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f16_bf16",   crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f16_i32",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f16_u32",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f16_i64",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f16_u8",     crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f16_f8e4m3", crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    // ----- cast_to_bf16: 9 sources (8-set + F8E4M3). -----
    cuda_ep!("cast_to_bf16_f32",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_bf16_f64",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_bf16_f16",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_bf16_bf16",   crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_bf16_i32",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_bf16_u32",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_bf16_i64",    crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_bf16_u8",     crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_bf16_f8e4m3", crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    // ----- cast_to_f64: 8 sources (8-set; no F8E4M3). -----
    cuda_ep!("cast_to_f64_f32",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f64_f64",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f64_f16",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f64_bf16", crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f64_i32",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f64_u32",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f64_i64",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f64_u8",   crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    // ----- cast_to_i32: 8 sources (8-set; no F8E4M3). -----
    cuda_ep!("cast_to_i32_f32",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_i32_f64",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_i32_f16",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_i32_bf16", crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_i32_i32",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_i32_u32",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_i32_i64",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_i32_u8",   crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    // ----- cast_to_u32: 8 sources (8-set; no F8E4M3). -----
    cuda_ep!("cast_to_u32_f32",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_u32_f64",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_u32_f16",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_u32_bf16", crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_u32_i32",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_u32_u32",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_u32_i64",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_u32_u8",   crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    // ----- cast_to_i64: 8 sources (8-set; no F8E4M3). -----
    cuda_ep!("cast_to_i64_f32",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_i64_f64",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_i64_f16",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_i64_bf16", crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_i64_i32",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_i64_u32",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_i64_i64",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_i64_u8",   crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    // ----- cast_to_u8: 8 sources (8-set; no F8E4M3). -----
    cuda_ep!("cast_to_u8_f32",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_u8_f64",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_u8_f16",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_u8_bf16", crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_u8_i32",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_u8_u32",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_u8_i64",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_u8_u8",   crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    // ----- cast_to_f8e4m3: 3 sources ({F32, F16, BF16}). -----
    cuda_ep!("cast_to_f8e4m3_f32",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f8e4m3_f16",  crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
    cuda_ep!("cast_to_f8e4m3_bf16", crate::baracuda_dispatch::cast::cast_baracuda_wrapper),
];

/// CUDA `binary` family (elementwise binary (add / sub / mul / div / maximum / minimum / pow / rem)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/binary.fkc.md`.
pub static CUDA_BINARY_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("add_f32", crate::baracuda_dispatch::binary::add_f32),
    cuda_ep!("add_f16", crate::baracuda_dispatch::binary::add_f16),
    cuda_ep!("add_bf16", crate::baracuda_dispatch::binary::add_bf16),
    cuda_ep!("add_f64", crate::baracuda_dispatch::binary::add_f64),
    cuda_ep!("sub_f32", crate::baracuda_dispatch::binary::sub_f32),
    cuda_ep!("sub_f16", crate::baracuda_dispatch::binary::sub_f16),
    cuda_ep!("sub_bf16", crate::baracuda_dispatch::binary::sub_bf16),
    cuda_ep!("sub_f64", crate::baracuda_dispatch::binary::sub_f64),
    cuda_ep!("mul_f32", crate::baracuda_dispatch::binary::mul_f32),
    cuda_ep!("mul_f16", crate::baracuda_dispatch::binary::mul_f16),
    cuda_ep!("mul_bf16", crate::baracuda_dispatch::binary::mul_bf16),
    cuda_ep!("mul_f64", crate::baracuda_dispatch::binary::mul_f64),
    cuda_ep!("div_f32", crate::baracuda_dispatch::binary::div_f32),
    cuda_ep!("div_f16", crate::baracuda_dispatch::binary::div_f16),
    cuda_ep!("div_bf16", crate::baracuda_dispatch::binary::div_bf16),
    cuda_ep!("div_f64", crate::baracuda_dispatch::binary::div_f64),
    cuda_ep!("maximum_f32", crate::baracuda_dispatch::binary::maximum_f32),
    cuda_ep!("maximum_f16", crate::baracuda_dispatch::binary::maximum_f16),
    cuda_ep!("maximum_bf16", crate::baracuda_dispatch::binary::maximum_bf16),
    cuda_ep!("maximum_f64", crate::baracuda_dispatch::binary::maximum_f64),
    cuda_ep!("minimum_f32", crate::baracuda_dispatch::binary::minimum_f32),
    cuda_ep!("minimum_f16", crate::baracuda_dispatch::binary::minimum_f16),
    cuda_ep!("minimum_bf16", crate::baracuda_dispatch::binary::minimum_bf16),
    cuda_ep!("minimum_f64", crate::baracuda_dispatch::binary::minimum_f64),
    cuda_ep!("pow_f32", crate::baracuda_dispatch::binary::pow_f32),
    cuda_ep!("pow_f16", crate::baracuda_dispatch::binary::pow_f16),
    cuda_ep!("pow_bf16", crate::baracuda_dispatch::binary::pow_bf16),
    cuda_ep!("pow_f64", crate::baracuda_dispatch::binary::pow_f64),
    cuda_ep!("rem_f32", crate::baracuda_dispatch::binary::rem_f32),
    cuda_ep!("rem_f16", crate::baracuda_dispatch::binary::rem_f16),
    cuda_ep!("rem_bf16", crate::baracuda_dispatch::binary::rem_bf16),
    cuda_ep!("rem_f64", crate::baracuda_dispatch::binary::rem_f64),
];

/// CUDA `reduce` family (axis reductions (sum / max / min / mean)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/reduce.fkc.md`.
pub static CUDA_REDUCE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("sum_f32", crate::baracuda_dispatch::reduce::sum_f32),
    cuda_ep!("sum_f16", crate::baracuda_dispatch::reduce::sum_f16),
    cuda_ep!("sum_bf16", crate::baracuda_dispatch::reduce::sum_bf16),
    cuda_ep!("sum_f64", crate::baracuda_dispatch::reduce::sum_f64),
    cuda_ep!("max_f32", crate::baracuda_dispatch::reduce::max_f32),
    cuda_ep!("max_f16", crate::baracuda_dispatch::reduce::max_f16),
    cuda_ep!("max_bf16", crate::baracuda_dispatch::reduce::max_bf16),
    cuda_ep!("max_f64", crate::baracuda_dispatch::reduce::max_f64),
    cuda_ep!("min_f32", crate::baracuda_dispatch::reduce::min_f32),
    cuda_ep!("min_f16", crate::baracuda_dispatch::reduce::min_f16),
    cuda_ep!("min_bf16", crate::baracuda_dispatch::reduce::min_bf16),
    cuda_ep!("min_f64", crate::baracuda_dispatch::reduce::min_f64),
    cuda_ep!("mean_f32", crate::baracuda_dispatch::reduce::mean_f32),
    cuda_ep!("mean_f16", crate::baracuda_dispatch::reduce::mean_f16),
    cuda_ep!("mean_bf16", crate::baracuda_dispatch::reduce::mean_bf16),
    cuda_ep!("mean_f64", crate::baracuda_dispatch::reduce::mean_f64),
];

/// CUDA `norm` family (last-dim normalizations (RmsNorm / LayerNorm)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/norm.fkc.md`.
pub static CUDA_NORM_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("rms_f32", crate::baracuda_dispatch::norm::rms_f32),
    cuda_ep!("rms_f16", crate::baracuda_dispatch::norm::rms_f16),
    cuda_ep!("rms_bf16", crate::baracuda_dispatch::norm::rms_bf16),
    cuda_ep!("rms_f64", crate::baracuda_dispatch::norm::rms_f64),
    cuda_ep!("layer_f32", crate::baracuda_dispatch::norm::layer_f32),
    cuda_ep!("layer_f16", crate::baracuda_dispatch::norm::layer_f16),
    cuda_ep!("layer_bf16", crate::baracuda_dispatch::norm::layer_bf16),
    cuda_ep!("layer_f64", crate::baracuda_dispatch::norm::layer_f64),
];

/// CUDA `softmax` family (last-dim softmax / log-softmax): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/softmax.fkc.md`.
pub static CUDA_SOFTMAX_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("softmax_f32", crate::baracuda_dispatch::softmax::softmax_f32),
    cuda_ep!("softmax_f16", crate::baracuda_dispatch::softmax::softmax_f16),
    cuda_ep!("softmax_bf16", crate::baracuda_dispatch::softmax::softmax_bf16),
    cuda_ep!("softmax_f64", crate::baracuda_dispatch::softmax::softmax_f64),
    cuda_ep!("log_softmax_f32", crate::baracuda_dispatch::softmax::log_softmax_f32),
    cuda_ep!("log_softmax_f16", crate::baracuda_dispatch::softmax::log_softmax_f16),
    cuda_ep!("log_softmax_bf16", crate::baracuda_dispatch::softmax::log_softmax_bf16),
    cuda_ep!("log_softmax_f64", crate::baracuda_dispatch::softmax::log_softmax_f64),
];

/// CUDA `powi` family (integer-exponent power (x^n)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/powi.fkc.md`.
pub static CUDA_POWI_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("powi_f32", crate::baracuda_dispatch::powi::powi_f32),
    cuda_ep!("powi_f64", crate::baracuda_dispatch::powi::powi_f64),
    cuda_ep!("powi_f16", crate::baracuda_dispatch::powi::powi_f16),
    cuda_ep!("powi_bf16", crate::baracuda_dispatch::powi::powi_bf16),
];

/// CUDA `powi_backward` family (integer-exponent power backward): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/powi_backward.fkc.md`.
pub static CUDA_POWI_BACKWARD_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("powi_backward_f32", crate::baracuda_dispatch::powi_backward::powi_backward_f32),
    cuda_ep!("powi_backward_f64", crate::baracuda_dispatch::powi_backward::powi_backward_f64),
    cuda_ep!("powi_backward_f16", crate::baracuda_dispatch::powi_backward::powi_backward_f16),
    cuda_ep!("powi_backward_bf16", crate::baracuda_dispatch::powi_backward::powi_backward_bf16),
];

/// CUDA `clamp` family (scalar-bounds clamp): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/clamp.fkc.md`.
pub static CUDA_CLAMP_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("clamp_f32", crate::baracuda_dispatch::clamp::clamp_f32),
    cuda_ep!("clamp_f64", crate::baracuda_dispatch::clamp::clamp_f64),
    cuda_ep!("clamp_f16", crate::baracuda_dispatch::clamp::clamp_f16),
    cuda_ep!("clamp_bf16", crate::baracuda_dispatch::clamp::clamp_bf16),
];

/// CUDA `flip` family (single-axis reverse (flip)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/flip.fkc.md`.
pub static CUDA_FLIP_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("flip_f32", crate::baracuda_dispatch::flip::flip_f32),
    cuda_ep!("flip_f16", crate::baracuda_dispatch::flip::flip_f16),
    cuda_ep!("flip_bf16", crate::baracuda_dispatch::flip::flip_bf16),
    cuda_ep!("flip_f64", crate::baracuda_dispatch::flip::flip_f64),
];

/// CUDA `roll` family (single-axis cyclic shift (roll)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/roll.fkc.md`.
pub static CUDA_ROLL_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("roll_f32", crate::baracuda_dispatch::roll::roll_f32),
    cuda_ep!("roll_f16", crate::baracuda_dispatch::roll::roll_f16),
    cuda_ep!("roll_bf16", crate::baracuda_dispatch::roll::roll_bf16),
    cuda_ep!("roll_f64", crate::baracuda_dispatch::roll::roll_f64),
];

/// CUDA `cumsum` family (single-axis inclusive prefix sum (cumsum)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/cumsum.fkc.md`.
pub static CUDA_CUMSUM_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("cumsum_f32", crate::baracuda_dispatch::cumsum::cumsum_f32),
    cuda_ep!("cumsum_f16", crate::baracuda_dispatch::cumsum::cumsum_f16),
    cuda_ep!("cumsum_bf16", crate::baracuda_dispatch::cumsum::cumsum_bf16),
    cuda_ep!("cumsum_f64", crate::baracuda_dispatch::cumsum::cumsum_f64),
];

/// CUDA `triangular` family (triangular masks (triu / tril)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/triangular.fkc.md`.
pub static CUDA_TRIANGULAR_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("triu_f32", crate::baracuda_dispatch::triangular::triu_f32),
    cuda_ep!("triu_f64", crate::baracuda_dispatch::triangular::triu_f64),
    cuda_ep!("triu_f16", crate::baracuda_dispatch::triangular::triu_f16),
    cuda_ep!("triu_bf16", crate::baracuda_dispatch::triangular::triu_bf16),
    cuda_ep!("triu_i32", crate::baracuda_dispatch::triangular::triu_i32),
    cuda_ep!("triu_i64", crate::baracuda_dispatch::triangular::triu_i64),
    cuda_ep!("tril_f32", crate::baracuda_dispatch::triangular::tril_f32),
    cuda_ep!("tril_f64", crate::baracuda_dispatch::triangular::tril_f64),
    cuda_ep!("tril_f16", crate::baracuda_dispatch::triangular::tril_f16),
    cuda_ep!("tril_bf16", crate::baracuda_dispatch::triangular::tril_bf16),
    cuda_ep!("tril_i32", crate::baracuda_dispatch::triangular::tril_i32),
    cuda_ep!("tril_i64", crate::baracuda_dispatch::triangular::tril_i64),
];

/// CUDA `arg_reduce` family (arg reductions (argmax / argmin over a dim -> U32)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/arg_reduce.fkc.md`.
pub static CUDA_ARG_REDUCE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("argmax_dim_u32_f32", crate::baracuda_dispatch::arg_reduce::argmax_dim_u32_f32),
    cuda_ep!("argmax_dim_u32_f64", crate::baracuda_dispatch::arg_reduce::argmax_dim_u32_f64),
    cuda_ep!("argmax_dim_u32_f16", crate::baracuda_dispatch::arg_reduce::argmax_dim_u32_f16),
    cuda_ep!("argmax_dim_u32_bf16", crate::baracuda_dispatch::arg_reduce::argmax_dim_u32_bf16),
    cuda_ep!("argmin_dim_u32_f32", crate::baracuda_dispatch::arg_reduce::argmin_dim_u32_f32),
    cuda_ep!("argmin_dim_u32_f64", crate::baracuda_dispatch::arg_reduce::argmin_dim_u32_f64),
    cuda_ep!("argmin_dim_u32_f16", crate::baracuda_dispatch::arg_reduce::argmin_dim_u32_f16),
    cuda_ep!("argmin_dim_u32_bf16", crate::baracuda_dispatch::arg_reduce::argmin_dim_u32_bf16),
];

/// CUDA `reduce_to` family (broadcast-reverse reductions (sum_to / max_to)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/reduce_to.fkc.md`.
pub static CUDA_REDUCE_TO_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("sum_to_f32", crate::baracuda_dispatch::reduce_to::sum_to_f32),
    cuda_ep!("sum_to_f16", crate::baracuda_dispatch::reduce_to::sum_to_f16),
    cuda_ep!("sum_to_bf16", crate::baracuda_dispatch::reduce_to::sum_to_bf16),
    cuda_ep!("sum_to_f64", crate::baracuda_dispatch::reduce_to::sum_to_f64),
    cuda_ep!("max_to_f32", crate::baracuda_dispatch::reduce_to::max_to_f32),
    cuda_ep!("max_to_f16", crate::baracuda_dispatch::reduce_to::max_to_f16),
    cuda_ep!("max_to_bf16", crate::baracuda_dispatch::reduce_to::max_to_bf16),
    cuda_ep!("max_to_f64", crate::baracuda_dispatch::reduce_to::max_to_f64),
];

/// CUDA `rope` family (rotary position embedding (RoPE)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/rope.fkc.md`.
pub static CUDA_ROPE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("rope_f32", crate::baracuda_dispatch::attention::rope_f32),
    cuda_ep!("rope_f16", crate::baracuda_dispatch::attention::rope_f16),
    cuda_ep!("rope_bf16", crate::baracuda_dispatch::attention::rope_bf16),
    cuda_ep!("rope_f64", crate::baracuda_dispatch::attention::rope_f64),
];

/// CUDA `unary` family (forward elementwise unary (21 ops x 4 dtypes + f32-only Sign)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/unary.fkc.md`.
pub static CUDA_UNARY_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("neg_f32", crate::baracuda_dispatch::unary::neg_f32),
    cuda_ep!("neg_f16", crate::baracuda_dispatch::unary::neg_f16),
    cuda_ep!("neg_bf16", crate::baracuda_dispatch::unary::neg_bf16),
    cuda_ep!("neg_f64", crate::baracuda_dispatch::unary::neg_f64),
    cuda_ep!("abs_f32", crate::baracuda_dispatch::unary::abs_f32),
    cuda_ep!("abs_f16", crate::baracuda_dispatch::unary::abs_f16),
    cuda_ep!("abs_bf16", crate::baracuda_dispatch::unary::abs_bf16),
    cuda_ep!("abs_f64", crate::baracuda_dispatch::unary::abs_f64),
    cuda_ep!("sqr_f32", crate::baracuda_dispatch::unary::sqr_f32),
    cuda_ep!("sqr_f16", crate::baracuda_dispatch::unary::sqr_f16),
    cuda_ep!("sqr_bf16", crate::baracuda_dispatch::unary::sqr_bf16),
    cuda_ep!("sqr_f64", crate::baracuda_dispatch::unary::sqr_f64),
    cuda_ep!("sqrt_f32", crate::baracuda_dispatch::unary::sqrt_f32),
    cuda_ep!("sqrt_f16", crate::baracuda_dispatch::unary::sqrt_f16),
    cuda_ep!("sqrt_bf16", crate::baracuda_dispatch::unary::sqrt_bf16),
    cuda_ep!("sqrt_f64", crate::baracuda_dispatch::unary::sqrt_f64),
    cuda_ep!("recip_f32", crate::baracuda_dispatch::unary::recip_f32),
    cuda_ep!("recip_f16", crate::baracuda_dispatch::unary::recip_f16),
    cuda_ep!("recip_bf16", crate::baracuda_dispatch::unary::recip_bf16),
    cuda_ep!("recip_f64", crate::baracuda_dispatch::unary::recip_f64),
    cuda_ep!("exp_f32", crate::baracuda_dispatch::unary::exp_f32),
    cuda_ep!("exp_f16", crate::baracuda_dispatch::unary::exp_f16),
    cuda_ep!("exp_bf16", crate::baracuda_dispatch::unary::exp_bf16),
    cuda_ep!("exp_f64", crate::baracuda_dispatch::unary::exp_f64),
    cuda_ep!("log_f32", crate::baracuda_dispatch::unary::log_f32),
    cuda_ep!("log_f16", crate::baracuda_dispatch::unary::log_f16),
    cuda_ep!("log_bf16", crate::baracuda_dispatch::unary::log_bf16),
    cuda_ep!("log_f64", crate::baracuda_dispatch::unary::log_f64),
    cuda_ep!("sin_f32", crate::baracuda_dispatch::unary::sin_f32),
    cuda_ep!("sin_f16", crate::baracuda_dispatch::unary::sin_f16),
    cuda_ep!("sin_bf16", crate::baracuda_dispatch::unary::sin_bf16),
    cuda_ep!("sin_f64", crate::baracuda_dispatch::unary::sin_f64),
    cuda_ep!("cos_f32", crate::baracuda_dispatch::unary::cos_f32),
    cuda_ep!("cos_f16", crate::baracuda_dispatch::unary::cos_f16),
    cuda_ep!("cos_bf16", crate::baracuda_dispatch::unary::cos_bf16),
    cuda_ep!("cos_f64", crate::baracuda_dispatch::unary::cos_f64),
    cuda_ep!("tanh_f32", crate::baracuda_dispatch::unary::tanh_f32),
    cuda_ep!("tanh_f16", crate::baracuda_dispatch::unary::tanh_f16),
    cuda_ep!("tanh_bf16", crate::baracuda_dispatch::unary::tanh_bf16),
    cuda_ep!("tanh_f64", crate::baracuda_dispatch::unary::tanh_f64),
    cuda_ep!("relu_f32", crate::baracuda_dispatch::unary::relu_f32),
    cuda_ep!("relu_f16", crate::baracuda_dispatch::unary::relu_f16),
    cuda_ep!("relu_bf16", crate::baracuda_dispatch::unary::relu_bf16),
    cuda_ep!("relu_f64", crate::baracuda_dispatch::unary::relu_f64),
    cuda_ep!("gelu_tanh_f32", crate::baracuda_dispatch::unary::gelu_tanh_f32),
    cuda_ep!("gelu_tanh_f16", crate::baracuda_dispatch::unary::gelu_tanh_f16),
    cuda_ep!("gelu_tanh_bf16", crate::baracuda_dispatch::unary::gelu_tanh_bf16),
    cuda_ep!("gelu_tanh_f64", crate::baracuda_dispatch::unary::gelu_tanh_f64),
    cuda_ep!("gelu_f32", crate::baracuda_dispatch::unary::gelu_f32),
    cuda_ep!("gelu_f16", crate::baracuda_dispatch::unary::gelu_f16),
    cuda_ep!("gelu_bf16", crate::baracuda_dispatch::unary::gelu_bf16),
    cuda_ep!("gelu_f64", crate::baracuda_dispatch::unary::gelu_f64),
    cuda_ep!("step_f32", crate::baracuda_dispatch::unary::step_f32),
    cuda_ep!("step_f16", crate::baracuda_dispatch::unary::step_f16),
    cuda_ep!("step_bf16", crate::baracuda_dispatch::unary::step_bf16),
    cuda_ep!("step_f64", crate::baracuda_dispatch::unary::step_f64),
    cuda_ep!("silu_f32", crate::baracuda_dispatch::unary::silu_f32),
    cuda_ep!("silu_f16", crate::baracuda_dispatch::unary::silu_f16),
    cuda_ep!("silu_bf16", crate::baracuda_dispatch::unary::silu_bf16),
    cuda_ep!("silu_f64", crate::baracuda_dispatch::unary::silu_f64),
    cuda_ep!("sigmoid_f32", crate::baracuda_dispatch::unary::sigmoid_f32),
    cuda_ep!("sigmoid_f16", crate::baracuda_dispatch::unary::sigmoid_f16),
    cuda_ep!("sigmoid_bf16", crate::baracuda_dispatch::unary::sigmoid_bf16),
    cuda_ep!("sigmoid_f64", crate::baracuda_dispatch::unary::sigmoid_f64),
    cuda_ep!("rsqrt_f32", crate::baracuda_dispatch::unary::rsqrt_f32),
    cuda_ep!("rsqrt_f16", crate::baracuda_dispatch::unary::rsqrt_f16),
    cuda_ep!("rsqrt_bf16", crate::baracuda_dispatch::unary::rsqrt_bf16),
    cuda_ep!("rsqrt_f64", crate::baracuda_dispatch::unary::rsqrt_f64),
    cuda_ep!("floor_f32", crate::baracuda_dispatch::unary::floor_f32),
    cuda_ep!("floor_f16", crate::baracuda_dispatch::unary::floor_f16),
    cuda_ep!("floor_bf16", crate::baracuda_dispatch::unary::floor_bf16),
    cuda_ep!("floor_f64", crate::baracuda_dispatch::unary::floor_f64),
    cuda_ep!("ceil_f32", crate::baracuda_dispatch::unary::ceil_f32),
    cuda_ep!("ceil_f16", crate::baracuda_dispatch::unary::ceil_f16),
    cuda_ep!("ceil_bf16", crate::baracuda_dispatch::unary::ceil_bf16),
    cuda_ep!("ceil_f64", crate::baracuda_dispatch::unary::ceil_f64),
    cuda_ep!("round_f32", crate::baracuda_dispatch::unary::round_f32),
    cuda_ep!("round_f16", crate::baracuda_dispatch::unary::round_f16),
    cuda_ep!("round_bf16", crate::baracuda_dispatch::unary::round_bf16),
    cuda_ep!("round_f64", crate::baracuda_dispatch::unary::round_f64),
    cuda_ep!("erf_f32", crate::baracuda_dispatch::unary::erf_f32),
    cuda_ep!("erf_f16", crate::baracuda_dispatch::unary::erf_f16),
    cuda_ep!("erf_bf16", crate::baracuda_dispatch::unary::erf_bf16),
    cuda_ep!("erf_f64", crate::baracuda_dispatch::unary::erf_f64),
    cuda_ep!("sign", crate::baracuda_dispatch::unary::sign_f32),
];

/// CUDA `clamp_inplace` family (in-place scalar-bounds clamp): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/clamp_inplace.fkc.md`.
pub static CUDA_CLAMP_INPLACE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("clamp_inplace_f32", crate::baracuda_dispatch::clamp_inplace::clamp_inplace_f32),
    cuda_ep!("clamp_inplace_f64", crate::baracuda_dispatch::clamp_inplace::clamp_inplace_f64),
    cuda_ep!("clamp_inplace_bf16", crate::baracuda_dispatch::clamp_inplace::clamp_inplace_bf16),
    cuda_ep!("clamp_inplace_f16", crate::baracuda_dispatch::clamp_inplace::clamp_inplace_f16),
];

/// CUDA `powi_inplace` family (in-place integer-exponent power): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/powi_inplace.fkc.md`.
pub static CUDA_POWI_INPLACE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("powi_inplace_f32", crate::baracuda_dispatch::powi_inplace::powi_inplace_f32),
    cuda_ep!("powi_inplace_f64", crate::baracuda_dispatch::powi_inplace::powi_inplace_f64),
    cuda_ep!("powi_inplace_bf16", crate::baracuda_dispatch::powi_inplace::powi_inplace_bf16),
    cuda_ep!("powi_inplace_f16", crate::baracuda_dispatch::powi_inplace::powi_inplace_f16),
];

/// CUDA `inplace_unary` family (in-place unary activations + the 16-op unary expansion): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/inplace_unary.fkc.md`.
pub static CUDA_INPLACE_UNARY_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("relu_inplace_f32", crate::baracuda_dispatch::unary::relu_inplace_f32),
    cuda_ep!("relu_inplace_f64", crate::baracuda_dispatch::unary::relu_inplace_f64),
    cuda_ep!("relu_inplace_bf16", crate::baracuda_dispatch::unary::relu_inplace_bf16),
    cuda_ep!("relu_inplace_f16", crate::baracuda_dispatch::unary::relu_inplace_f16),
    cuda_ep!("silu_inplace_f32", crate::baracuda_dispatch::unary::silu_inplace_f32),
    cuda_ep!("silu_inplace_f64", crate::baracuda_dispatch::unary::silu_inplace_f64),
    cuda_ep!("silu_inplace_bf16", crate::baracuda_dispatch::unary::silu_inplace_bf16),
    cuda_ep!("silu_inplace_f16", crate::baracuda_dispatch::unary::silu_inplace_f16),
    cuda_ep!("gelu_inplace_f32", crate::baracuda_dispatch::unary::gelu_inplace_f32),
    cuda_ep!("gelu_inplace_f64", crate::baracuda_dispatch::unary::gelu_inplace_f64),
    cuda_ep!("gelu_inplace_bf16", crate::baracuda_dispatch::unary::gelu_inplace_bf16),
    cuda_ep!("gelu_inplace_f16", crate::baracuda_dispatch::unary::gelu_inplace_f16),
    cuda_ep!("tanh_inplace_f32", crate::baracuda_dispatch::unary::tanh_inplace_f32),
    cuda_ep!("tanh_inplace_f64", crate::baracuda_dispatch::unary::tanh_inplace_f64),
    cuda_ep!("tanh_inplace_bf16", crate::baracuda_dispatch::unary::tanh_inplace_bf16),
    cuda_ep!("tanh_inplace_f16", crate::baracuda_dispatch::unary::tanh_inplace_f16),
    cuda_ep!("sigmoid_inplace_f32", crate::baracuda_dispatch::unary::sigmoid_inplace_f32),
    cuda_ep!("sigmoid_inplace_f64", crate::baracuda_dispatch::unary::sigmoid_inplace_f64),
    cuda_ep!("sigmoid_inplace_bf16", crate::baracuda_dispatch::unary::sigmoid_inplace_bf16),
    cuda_ep!("sigmoid_inplace_f16", crate::baracuda_dispatch::unary::sigmoid_inplace_f16),
    cuda_ep!("neg_inplace_f32", crate::baracuda_dispatch::unary::neg_inplace_f32),
    cuda_ep!("neg_inplace_f64", crate::baracuda_dispatch::unary::neg_inplace_f64),
    cuda_ep!("neg_inplace_bf16", crate::baracuda_dispatch::unary::neg_inplace_bf16),
    cuda_ep!("neg_inplace_f16", crate::baracuda_dispatch::unary::neg_inplace_f16),
    cuda_ep!("abs_inplace_f32", crate::baracuda_dispatch::unary::abs_inplace_f32),
    cuda_ep!("abs_inplace_f64", crate::baracuda_dispatch::unary::abs_inplace_f64),
    cuda_ep!("abs_inplace_bf16", crate::baracuda_dispatch::unary::abs_inplace_bf16),
    cuda_ep!("abs_inplace_f16", crate::baracuda_dispatch::unary::abs_inplace_f16),
    cuda_ep!("sqr_inplace_f32", crate::baracuda_dispatch::unary::sqr_inplace_f32),
    cuda_ep!("sqr_inplace_f64", crate::baracuda_dispatch::unary::sqr_inplace_f64),
    cuda_ep!("sqr_inplace_bf16", crate::baracuda_dispatch::unary::sqr_inplace_bf16),
    cuda_ep!("sqr_inplace_f16", crate::baracuda_dispatch::unary::sqr_inplace_f16),
    cuda_ep!("sqrt_inplace_f32", crate::baracuda_dispatch::unary::sqrt_inplace_f32),
    cuda_ep!("sqrt_inplace_f64", crate::baracuda_dispatch::unary::sqrt_inplace_f64),
    cuda_ep!("sqrt_inplace_bf16", crate::baracuda_dispatch::unary::sqrt_inplace_bf16),
    cuda_ep!("sqrt_inplace_f16", crate::baracuda_dispatch::unary::sqrt_inplace_f16),
    cuda_ep!("rsqrt_inplace_f32", crate::baracuda_dispatch::unary::rsqrt_inplace_f32),
    cuda_ep!("rsqrt_inplace_f64", crate::baracuda_dispatch::unary::rsqrt_inplace_f64),
    cuda_ep!("rsqrt_inplace_bf16", crate::baracuda_dispatch::unary::rsqrt_inplace_bf16),
    cuda_ep!("rsqrt_inplace_f16", crate::baracuda_dispatch::unary::rsqrt_inplace_f16),
    cuda_ep!("recip_inplace_f32", crate::baracuda_dispatch::unary::recip_inplace_f32),
    cuda_ep!("recip_inplace_f64", crate::baracuda_dispatch::unary::recip_inplace_f64),
    cuda_ep!("recip_inplace_bf16", crate::baracuda_dispatch::unary::recip_inplace_bf16),
    cuda_ep!("recip_inplace_f16", crate::baracuda_dispatch::unary::recip_inplace_f16),
    cuda_ep!("exp_inplace_f32", crate::baracuda_dispatch::unary::exp_inplace_f32),
    cuda_ep!("exp_inplace_f64", crate::baracuda_dispatch::unary::exp_inplace_f64),
    cuda_ep!("exp_inplace_bf16", crate::baracuda_dispatch::unary::exp_inplace_bf16),
    cuda_ep!("exp_inplace_f16", crate::baracuda_dispatch::unary::exp_inplace_f16),
    cuda_ep!("log_inplace_f32", crate::baracuda_dispatch::unary::log_inplace_f32),
    cuda_ep!("log_inplace_f64", crate::baracuda_dispatch::unary::log_inplace_f64),
    cuda_ep!("log_inplace_bf16", crate::baracuda_dispatch::unary::log_inplace_bf16),
    cuda_ep!("log_inplace_f16", crate::baracuda_dispatch::unary::log_inplace_f16),
    cuda_ep!("sin_inplace_f32", crate::baracuda_dispatch::unary::sin_inplace_f32),
    cuda_ep!("sin_inplace_f64", crate::baracuda_dispatch::unary::sin_inplace_f64),
    cuda_ep!("sin_inplace_bf16", crate::baracuda_dispatch::unary::sin_inplace_bf16),
    cuda_ep!("sin_inplace_f16", crate::baracuda_dispatch::unary::sin_inplace_f16),
    cuda_ep!("cos_inplace_f32", crate::baracuda_dispatch::unary::cos_inplace_f32),
    cuda_ep!("cos_inplace_f64", crate::baracuda_dispatch::unary::cos_inplace_f64),
    cuda_ep!("cos_inplace_bf16", crate::baracuda_dispatch::unary::cos_inplace_bf16),
    cuda_ep!("cos_inplace_f16", crate::baracuda_dispatch::unary::cos_inplace_f16),
    cuda_ep!("sign_inplace_f32", crate::baracuda_dispatch::unary::sign_inplace_f32),
    cuda_ep!("sign_inplace_f64", crate::baracuda_dispatch::unary::sign_inplace_f64),
    cuda_ep!("sign_inplace_bf16", crate::baracuda_dispatch::unary::sign_inplace_bf16),
    cuda_ep!("sign_inplace_f16", crate::baracuda_dispatch::unary::sign_inplace_f16),
    cuda_ep!("floor_inplace_f32", crate::baracuda_dispatch::unary::floor_inplace_f32),
    cuda_ep!("floor_inplace_f64", crate::baracuda_dispatch::unary::floor_inplace_f64),
    cuda_ep!("floor_inplace_bf16", crate::baracuda_dispatch::unary::floor_inplace_bf16),
    cuda_ep!("floor_inplace_f16", crate::baracuda_dispatch::unary::floor_inplace_f16),
    cuda_ep!("ceil_inplace_f32", crate::baracuda_dispatch::unary::ceil_inplace_f32),
    cuda_ep!("ceil_inplace_f64", crate::baracuda_dispatch::unary::ceil_inplace_f64),
    cuda_ep!("ceil_inplace_bf16", crate::baracuda_dispatch::unary::ceil_inplace_bf16),
    cuda_ep!("ceil_inplace_f16", crate::baracuda_dispatch::unary::ceil_inplace_f16),
    cuda_ep!("round_inplace_f32", crate::baracuda_dispatch::unary::round_inplace_f32),
    cuda_ep!("round_inplace_f64", crate::baracuda_dispatch::unary::round_inplace_f64),
    cuda_ep!("round_inplace_bf16", crate::baracuda_dispatch::unary::round_inplace_bf16),
    cuda_ep!("round_inplace_f16", crate::baracuda_dispatch::unary::round_inplace_f16),
    cuda_ep!("erf_inplace_f32", crate::baracuda_dispatch::unary::erf_inplace_f32),
    cuda_ep!("erf_inplace_f64", crate::baracuda_dispatch::unary::erf_inplace_f64),
    cuda_ep!("erf_inplace_bf16", crate::baracuda_dispatch::unary::erf_inplace_bf16),
    cuda_ep!("erf_inplace_f16", crate::baracuda_dispatch::unary::erf_inplace_f16),
    cuda_ep!("gelu_erf_inplace_f32", crate::baracuda_dispatch::unary::gelu_erf_inplace_f32),
    cuda_ep!("gelu_erf_inplace_f64", crate::baracuda_dispatch::unary::gelu_erf_inplace_f64),
    cuda_ep!("gelu_erf_inplace_bf16", crate::baracuda_dispatch::unary::gelu_erf_inplace_bf16),
    cuda_ep!("gelu_erf_inplace_f16", crate::baracuda_dispatch::unary::gelu_erf_inplace_f16),
];

/// CUDA `write_slice` family (in-place rectangular slab assign (WriteSlice; byte-width umbrella)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/write_slice.fkc.md`.
pub static CUDA_WRITE_SLICE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("write_slice_f32", crate::baracuda_dispatch::write_slice::write_slice_b4),
    cuda_ep!("write_slice_f64", crate::baracuda_dispatch::write_slice::write_slice_b8),
    cuda_ep!("write_slice_f16", crate::baracuda_dispatch::write_slice::write_slice_b2),
    cuda_ep!("write_slice_bf16", crate::baracuda_dispatch::write_slice::write_slice_b2),
    cuda_ep!("write_slice_i32", crate::baracuda_dispatch::write_slice::write_slice_b4),
    cuda_ep!("write_slice_i64", crate::baracuda_dispatch::write_slice::write_slice_b8),
    cuda_ep!("write_slice_u32", crate::baracuda_dispatch::write_slice::write_slice_b4),
    cuda_ep!("write_slice_u8", crate::baracuda_dispatch::write_slice::write_slice_b1),
    cuda_ep!("write_slice_i8", crate::baracuda_dispatch::write_slice::write_slice_b1),
];

/// CUDA `write_slice_rotating` family (sliding-window in-place slab assign (WriteSliceRotating; byte-width umbrella)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/write_slice_rotating.fkc.md`.
pub static CUDA_WRITE_SLICE_ROTATING_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("write_slice_rotating_f32", crate::baracuda_dispatch::write_slice_rotating::write_slice_rotating_b4),
    cuda_ep!("write_slice_rotating_f64", crate::baracuda_dispatch::write_slice_rotating::write_slice_rotating_b8),
    cuda_ep!("write_slice_rotating_f16", crate::baracuda_dispatch::write_slice_rotating::write_slice_rotating_b2),
    cuda_ep!("write_slice_rotating_bf16", crate::baracuda_dispatch::write_slice_rotating::write_slice_rotating_b2),
    cuda_ep!("write_slice_rotating_i32", crate::baracuda_dispatch::write_slice_rotating::write_slice_rotating_b4),
    cuda_ep!("write_slice_rotating_i64", crate::baracuda_dispatch::write_slice_rotating::write_slice_rotating_b8),
    cuda_ep!("write_slice_rotating_u32", crate::baracuda_dispatch::write_slice_rotating::write_slice_rotating_b4),
    cuda_ep!("write_slice_rotating_u8", crate::baracuda_dispatch::write_slice_rotating::write_slice_rotating_b1),
    cuda_ep!("write_slice_rotating_i8", crate::baracuda_dispatch::write_slice_rotating::write_slice_rotating_b1),
];

/// CUDA `write_slice_doff` family (device-resident-offset in-place slab assign (WriteSliceDoff; byte-width umbrella)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/write_slice_doff.fkc.md`. b1/b2/b4/b8 only (no b16 — the KV-decode dtype set).
pub static CUDA_WRITE_SLICE_DOFF_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("write_slice_doff_f32", crate::baracuda_dispatch::write_slice_doff::write_slice_doff_b4),
    cuda_ep!("write_slice_doff_f64", crate::baracuda_dispatch::write_slice_doff::write_slice_doff_b8),
    cuda_ep!("write_slice_doff_f16", crate::baracuda_dispatch::write_slice_doff::write_slice_doff_b2),
    cuda_ep!("write_slice_doff_bf16", crate::baracuda_dispatch::write_slice_doff::write_slice_doff_b2),
    cuda_ep!("write_slice_doff_i32", crate::baracuda_dispatch::write_slice_doff::write_slice_doff_b4),
    cuda_ep!("write_slice_doff_i64", crate::baracuda_dispatch::write_slice_doff::write_slice_doff_b8),
    cuda_ep!("write_slice_doff_u32", crate::baracuda_dispatch::write_slice_doff::write_slice_doff_b4),
    cuda_ep!("write_slice_doff_u8", crate::baracuda_dispatch::write_slice_doff::write_slice_doff_b1),
    cuda_ep!("write_slice_doff_i8", crate::baracuda_dispatch::write_slice_doff::write_slice_doff_b1),
];

/// CUDA `concat` family (N-ary concatenation (Concat; binding [T,T])): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/concat.fkc.md`.
pub static CUDA_CONCAT_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("concat_f32", crate::baracuda_dispatch::concat::concat_f32),
    cuda_ep!("concat_f16", crate::baracuda_dispatch::concat::concat_f16),
    cuda_ep!("concat_bf16", crate::baracuda_dispatch::concat::concat_bf16),
    cuda_ep!("concat_f64", crate::baracuda_dispatch::concat::concat_f64),
];

/// CUDA `affine` family (affine y = mul*x + add (Affine)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/affine.fkc.md`.
pub static CUDA_AFFINE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("affine_f32", crate::baracuda_dispatch::affine::affine_f32),
    cuda_ep!("affine_f64", crate::baracuda_dispatch::affine::affine_f64),
    cuda_ep!("affine_f16", crate::baracuda_dispatch::affine::affine_f16),
    cuda_ep!("affine_bf16", crate::baracuda_dispatch::affine::affine_bf16),
    cuda_ep!("affine_i32", crate::baracuda_dispatch::affine::affine_i32),
    cuda_ep!("affine_i64", crate::baracuda_dispatch::affine::affine_i64),
    cuda_ep!("affine_u8", crate::baracuda_dispatch::affine::affine_u8),
];

/// CUDA `inplace_affine` family (in-place affine (InplaceAffine)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/inplace_affine.fkc.md`.
pub static CUDA_INPLACE_AFFINE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("affine_inplace_f32", crate::baracuda_dispatch::affine::affine_inplace_f32),
    cuda_ep!("affine_inplace_f64", crate::baracuda_dispatch::affine::affine_inplace_f64),
    cuda_ep!("affine_inplace_bf16", crate::baracuda_dispatch::affine::affine_inplace_bf16),
    cuda_ep!("affine_inplace_f16", crate::baracuda_dispatch::affine::affine_inplace_f16),
];

/// CUDA `pad` family (multi-dim padding (Pad)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/pad.fkc.md`.
pub static CUDA_PAD_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("pad_f32", crate::baracuda_dispatch::pad::pad_f32),
    cuda_ep!("pad_f16", crate::baracuda_dispatch::pad::pad_f16),
    cuda_ep!("pad_bf16", crate::baracuda_dispatch::pad::pad_bf16),
    cuda_ep!("pad_f64", crate::baracuda_dispatch::pad::pad_f64),
];

/// CUDA `pad_backward` family (padding backward (PadBackward; Constant)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/pad_backward.fkc.md`.
pub static CUDA_PAD_BACKWARD_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("pad_backward_f32", crate::baracuda_dispatch::pad::pad_backward_f32),
    cuda_ep!("pad_backward_f16", crate::baracuda_dispatch::pad::pad_backward_f16),
    cuda_ep!("pad_backward_bf16", crate::baracuda_dispatch::pad::pad_backward_bf16),
    cuda_ep!("pad_backward_f64", crate::baracuda_dispatch::pad::pad_backward_f64),
];

/// CUDA `causal_conv1d` family (causal depthwise conv1d (CausalConv1d; 4-input key)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/causal_conv1d.fkc.md`.
pub static CUDA_CAUSAL_CONV1D_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("causal_conv1d_f32", crate::baracuda_dispatch::conv1d::causal_conv1d_f32),
    cuda_ep!("causal_conv1d_f64", crate::baracuda_dispatch::conv1d::causal_conv1d_f64),
    cuda_ep!("causal_conv1d_bf16", crate::baracuda_dispatch::conv1d::causal_conv1d_bf16),
    cuda_ep!("causal_conv1d_f16", crate::baracuda_dispatch::conv1d::causal_conv1d_f16),
];

/// CUDA `gemm_dense` family (dense FP matmul facade (gemm_dense at OpKind::MatMul)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/gemm_dense.fkc.md`.
pub static CUDA_GEMM_DENSE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("matmul_f32", crate::baracuda_dispatch::gemm_dense::matmul_f32),
    cuda_ep!("matmul_f16", crate::baracuda_dispatch::gemm_dense::matmul_f16),
    cuda_ep!("matmul_bf16", crate::baracuda_dispatch::gemm_dense::matmul_bf16),
    cuda_ep!("matmul_f64", crate::baracuda_dispatch::gemm_dense::matmul_f64),
];

/// CUDA `gemm_int` family (int8 matmul facade (gemm_int s8/u8 RRR at OpKind::MatMul)): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/gemm_int.fkc.md`.
pub static CUDA_GEMM_INT_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("gemm_i8", crate::baracuda_dispatch::gemm_int::gemm_s8_rrr),
    cuda_ep!("gemm_u8", crate::baracuda_dispatch::gemm_int::gemm_u8_rrr),
];

/// CUDA `indexing` family (IndexSelect / Gather / MaskedFill data-dtype fan +
/// ScatterAdd per-dtype): fanned `<op>_<dtype>` symbol -> production wrapper.
/// Contract: `docs/kernel-contracts/cuda/indexing.fkc.md`. Unlike the CPU
/// indexing family (a dtype-agnostic byte-copy umbrella), each CUDA symbol
/// resolves to its OWN per-dtype baracuda wrapper (the binary/unary precedent).
/// The `U32` index / `U8` mask slot is a FIXED single-dtype operand (not a fan
/// axis) — the compare-mask / paged block-table precedent — so keys are
/// `[T, U32, T]` (IndexSelect/Gather), `[T, U8, T]` (MaskedFill), and
/// `[T, U32, T, T]` (ScatterAdd). Contiguous-only (default caps ⇒
/// `strided_input == false`), byte-for-byte the deleted hand-written
/// `table.register(...)` regs.
pub static CUDA_INDEXING_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // IndexSelect — data-dtype fan {F32, F64, I32}; distinct per-dtype wrappers.
    cuda_ep!("index_select_f32", crate::baracuda_dispatch::indexing::index_select_f32),
    cuda_ep!("index_select_f64", crate::baracuda_dispatch::indexing::index_select_f64),
    cuda_ep!("index_select_i32", crate::baracuda_dispatch::indexing::index_select_i32),
    // Gather — data-dtype fan {F32, F64, I32}; distinct per-dtype wrappers.
    cuda_ep!("gather_f32", crate::baracuda_dispatch::indexing::gather_f32),
    cuda_ep!("gather_f64", crate::baracuda_dispatch::indexing::gather_f64),
    cuda_ep!("gather_i32", crate::baracuda_dispatch::indexing::gather_i32),
    // MaskedFill — data-dtype fan {F32, F64, I32}; fixed U8 mask slot.
    cuda_ep!("masked_fill_f32", crate::baracuda_dispatch::indexing::masked_fill_f32),
    cuda_ep!("masked_fill_f64", crate::baracuda_dispatch::indexing::masked_fill_f64),
    cuda_ep!("masked_fill_i32", crate::baracuda_dispatch::indexing::masked_fill_i32),
    // ScatterAdd — per-dtype {F32, F64}; single-dtype entry_point resolved AS-IS.
    cuda_ep!("scatter_add_f32", crate::baracuda_dispatch::indexing::scatter_add_f32),
    cuda_ep!("scatter_add_f64", crate::baracuda_dispatch::indexing::scatter_add_f64),
];

/// CUDA `flash_decoding` family (FlashDecoding decode arm; `OpKind::FlashAttn`,
/// `seq_q==1`, capacity-K): fanned `flash_decoding_<dtype>` symbol -> production
/// wrapper. Contract: `docs/kernel-contracts/cuda/flash_decoding.fkc.md`. The
/// NO-alibi decode arm keyed `[q, k, v, out] = [T; 4]` over `{F16, BF16}` —
/// byte-for-byte the two DELETED hand-written `register_full(FlashAttn, …)` regs.
/// Its cost is CONTRACT-PINNED via [`CUDA_COST_FNS`] (the §4.4 trampoline), NOT
/// the `fill_unset` default — see the module doc + `cost_flash_decoding_cuda`.
pub static CUDA_FLASH_DECODING_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    cuda_ep!("flash_decoding_f16",  crate::baracuda_dispatch::flash_decoding::flash_decoding_f16),
    cuda_ep!("flash_decoding_bf16", crate::baracuda_dispatch::flash_decoding::flash_decoding_bf16),
];

/// The built-in CUDA (baracuda) backend's NAMED cost-fn table (§4.4 cost-fn
/// trampoline, Task-F): maps a contract's `cost.cost_fn` NAME to the production
/// [`CostFn`] pointer. The cost-model analogue of the `CUDA_*_ENTRY_POINTS`
/// `entry_point` tables — resolved by [`CudaLinkRegistry::resolve_cost_fn`] so a
/// contract can PIN a real, shape-aware cost model that SURVIVES the
/// `fill_unset_cost_for_backend` pass (which only replaces the `unknown_cost`
/// sentinel). Today it carries the ONE cost fn a CUDA contract pins:
/// `flash_decoding`'s [`crate::cost::cost_flash_decoding_cuda`] — the static
/// infeasibility gate that returns an INFEASIBLE cost for `seq_q != 1` /
/// `head_dim` outside `[1, 128]`, keeping the ranker off unsupported shapes. A
/// contract that names a cost fn ABSENT here fails import with
/// [`crate::fkc::FkcError::UnknownCostFn`] (never a silent `unknown_cost`
/// fallback, never a fabricated pointer — the cost-model P9 analogue).
pub static CUDA_COST_FNS: &[(&str, CostFn)] = &[
    ("cost_flash_decoding_cuda", crate::cost::cost_flash_decoding_cuda as CostFn),
];

/// The built-in CUDA (baracuda) backend's [`LinkRegistry`] — resolves a
/// contract's `entry_point` symbols against [`CUDA_CAST_ENTRY_POINTS`] (and the
/// other migrated CUDA families) and its `cost.cost_fn` names against
/// [`CUDA_COST_FNS`]. Unresolved → `None`, which the importer turns into a typed
/// `UnknownEntryPoint` / `UnknownCostFn` error (never a panic, never a
/// fabricated pointer). Mirrors [`crate::fkc::VulkanLinkRegistry`].
pub struct CudaLinkRegistry;

impl LinkRegistry for CudaLinkRegistry {
    fn resolve_primitive(&self, symbol: &str) -> Option<KernelRef> {
        // Chain every migrated CUDA family's table; each symbol is unique
        // across families (cast_to_*, and the future per-op elementwise /
        // matmul / … symbols), so order is immaterial.
        CUDA_CAST_ENTRY_POINTS
            .iter()
            .chain(CUDA_BINARY_ENTRY_POINTS.iter())
            .chain(CUDA_REDUCE_ENTRY_POINTS.iter())
            .chain(CUDA_NORM_ENTRY_POINTS.iter())
            .chain(CUDA_SOFTMAX_ENTRY_POINTS.iter())
            .chain(CUDA_POWI_ENTRY_POINTS.iter())
            .chain(CUDA_POWI_BACKWARD_ENTRY_POINTS.iter())
            .chain(CUDA_CLAMP_ENTRY_POINTS.iter())
            .chain(CUDA_FLIP_ENTRY_POINTS.iter())
            .chain(CUDA_ROLL_ENTRY_POINTS.iter())
            .chain(CUDA_CUMSUM_ENTRY_POINTS.iter())
            .chain(CUDA_TRIANGULAR_ENTRY_POINTS.iter())
            .chain(CUDA_ARG_REDUCE_ENTRY_POINTS.iter())
            .chain(CUDA_REDUCE_TO_ENTRY_POINTS.iter())
            .chain(CUDA_ROPE_ENTRY_POINTS.iter())
            .chain(CUDA_UNARY_ENTRY_POINTS.iter())
            .chain(CUDA_CLAMP_INPLACE_ENTRY_POINTS.iter())
            .chain(CUDA_POWI_INPLACE_ENTRY_POINTS.iter())
            .chain(CUDA_INPLACE_UNARY_ENTRY_POINTS.iter())
            .chain(CUDA_WRITE_SLICE_ENTRY_POINTS.iter())
            .chain(CUDA_WRITE_SLICE_ROTATING_ENTRY_POINTS.iter())
            .chain(CUDA_WRITE_SLICE_DOFF_ENTRY_POINTS.iter())
            .chain(CUDA_CONCAT_ENTRY_POINTS.iter())
            .chain(CUDA_AFFINE_ENTRY_POINTS.iter())
            .chain(CUDA_INPLACE_AFFINE_ENTRY_POINTS.iter())
            .chain(CUDA_PAD_ENTRY_POINTS.iter())
            .chain(CUDA_PAD_BACKWARD_ENTRY_POINTS.iter())
            .chain(CUDA_CAUSAL_CONV1D_ENTRY_POINTS.iter())
            .chain(CUDA_GEMM_DENSE_ENTRY_POINTS.iter())
            .chain(CUDA_GEMM_INT_ENTRY_POINTS.iter())
            .chain(CUDA_INDEXING_ENTRY_POINTS.iter())
            .chain(CUDA_FLASH_DECODING_ENTRY_POINTS.iter())
            .find(|(s, _)| *s == symbol)
            .map(|(_, k)| *k)
    }

    fn resolve_fused(&self, _symbol: &str) -> Option<KernelRef> {
        // No fused-op contracts in the CUDA cast corpus — every section is a
        // primitive `op_kind`.
        None
    }

    fn resolve_cost_fn(&self, name: &str) -> Option<CostFn> {
        // §4.4 cost-fn trampoline: resolve a contract's `cost.cost_fn` NAME
        // against the CUDA named cost-fn table (the cost-model analogue of the
        // entry_point resolution above). Unresolved → None → typed
        // `UnknownCostFn` at the importer.
        CUDA_COST_FNS.iter().find(|(s, _)| *s == name).map(|(_, f)| *f)
    }
}
