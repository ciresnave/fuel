//! The built-in Vulkan backend's FKC `link_registry` (kernel-seam-interop §3.5,
//! §4.3; FKC §12.6). The GPU analogue of [`crate::fkc::CpuLinkRegistry`]: maps
//! each Vulkan kernel contract's `entry_point` symbol to the production dispatch
//! wrapper — the real, non-stub resolution the importer uses so an imported
//! contract binds the **actual** kernel (no raw pointers in the serialized
//! contract, FKC P9).
//!
//! For the built-in Vulkan backend the wrappers live in
//! [`crate::vulkan_dispatch`] (the dispatch layer that adapts the Slang/SPIR-V
//! byte-kernels to [`KernelRef`]) and this table co-locates with them in
//! fuel-dispatch. An *external* provider instead exports its own link registry
//! across the FFI; this is Fuel's internal Vulkan-provider analogue.
//!
//! Everything here is gated behind the `vulkan` cargo feature — the DEFAULT
//! build never compiles this module.

use crate::fkc::lower::LinkRegistry;
use crate::kernel::KernelRef;

/// One `(contract entry_point symbol, production wrapper)` pair. The symbol
/// matches the contract's `entry_point: "fuel_vulkan_backend::fkc::<name>"`.
/// Unlike the CPU `ep!` macro (which builds `<op>_<dt>` from two literals),
/// Vulkan cast sections name their symbol in full (`cast_<src>_to_<dst>`), so
/// this macro takes the bare symbol name + the fully-qualified wrapper path.
macro_rules! vk_ep {
    ($sym:literal, $wrapper:path $(,)?) => {
        (
            concat!("fuel_vulkan_backend::fkc::", $sym),
            $wrapper as KernelRef,
        )
    };
}

/// The Vulkan cast (dtype-conversion) family's `symbol → production wrapper`
/// map — the FULL family: 12 (SRC, DST) pairs resolving to 3 production
/// wrappers. Contract: `docs/kernel-contracts/vulkan/cast.fkc.md`. Each per-pair
/// section (`## cast_f32_to_f16`, …) declares a SPECIFIC single-dtype `src`
/// input + a `fixed(DST)` output, so none of them dtype-fan — the importer
/// resolves that symbol AS-IS and keys `[SRC, DST]` (input dtype + the output's
/// fixed dtype), byte-for-byte the deleted hand-written
/// `table.register_with_precision(OpKind::Cast, &[src, dst], …)` regs. The
/// `OpParams::Cast` variant is a unit marker (the target dtype rides on the
/// output Storage), so it never enters the key.
///
/// Several sections share ONE wrapper (as the deleted hand-written regs did):
/// - `cast::cast_f32_half` — the pair-packed half casts `f32↔f16` / `f32↔bf16`.
/// - `cast::cast_f32_f64` — the one-per-element wide casts `f32↔f64`.
/// - `cast_f8e4m3::cast_f8e4m3` — the six byte-packed `F8E4M3↔{f32,f16,bf16}`
///   casts (all non-F8 sides routed via F32 internally).
///
/// This contract has NO `##` chassis umbrella section, so there is no
/// `registrable: false` describe-only entry to omit.
pub static VULKAN_CAST_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // Pair-packed half casts (f32↔f16, f32↔bf16) → cast_f32_half.
    vk_ep!("cast_f32_to_f16",  crate::vulkan_dispatch::cast::cast_f32_half),
    vk_ep!("cast_f16_to_f32",  crate::vulkan_dispatch::cast::cast_f32_half),
    vk_ep!("cast_f32_to_bf16", crate::vulkan_dispatch::cast::cast_f32_half),
    vk_ep!("cast_bf16_to_f32", crate::vulkan_dispatch::cast::cast_f32_half),
    // One-per-element wide casts (f32↔f64) → cast_f32_f64.
    vk_ep!("cast_f32_to_f64",  crate::vulkan_dispatch::cast::cast_f32_f64),
    vk_ep!("cast_f64_to_f32",  crate::vulkan_dispatch::cast::cast_f32_f64),
    // Byte-packed F8E4M3 casts (F8E4M3 ↔ {f32, f16, bf16}) → cast_f8e4m3.
    vk_ep!("cast_f32_to_f8e4m3",  crate::vulkan_dispatch::cast_f8e4m3::cast_f8e4m3),
    vk_ep!("cast_f8e4m3_to_f32",  crate::vulkan_dispatch::cast_f8e4m3::cast_f8e4m3),
    vk_ep!("cast_f16_to_f8e4m3",  crate::vulkan_dispatch::cast_f8e4m3::cast_f8e4m3),
    vk_ep!("cast_f8e4m3_to_f16",  crate::vulkan_dispatch::cast_f8e4m3::cast_f8e4m3),
    vk_ep!("cast_bf16_to_f8e4m3", crate::vulkan_dispatch::cast_f8e4m3::cast_f8e4m3),
    vk_ep!("cast_f8e4m3_to_bf16", crate::vulkan_dispatch::cast_f8e4m3::cast_f8e4m3),
];

/// The Vulkan elementwise family's `symbol → production wrapper` map — the
/// FULL family: 94 (op, dtype) keys resolving to the per-(op, dtype) wrappers in
/// [`crate::vulkan_dispatch`]. Contract:
/// `docs/kernel-contracts/vulkan/elementwise.fkc.md`, re-authored per-op (the
/// CPU inplace precedent). The strided unary / affine sections declare a BASE
/// `entry_point` (`…::<op>` / `…::affine`) + `dtypes: [F32, F16, F64]`, so the
/// importer fans them into `<base>_{f32,f16,f64}` — the symbols below. The
/// binary sections fan `[F32, F16, F64, BF16]` (all four strided, incl. the
/// lane-masked `binary_bf16`). The bf16-unary / `affine_bf16` / `clamp` / `powi`
/// sections are single-dtype (resolved AS-IS on their full `<op>_<dt>` symbol).
///
/// Caps ride through the import truthfully (§6): the strided sections' layout
/// (`strided: accepted, broadcast_stride0: accepted`) projects
/// `strided_input=true` (byte-for-byte the deleted
/// `register_with_caps_and_precision(strided)` regs); the bf16-unary /
/// `affine_bf16` sections' `contiguous: required` layout projects
/// `strided_input=false` (the deleted plain `register_with_precision` regs).
pub static VULKAN_ELEMENTWISE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // ----- Unary: 16 ops, strided f32/f16/f64 fan + contiguous bf16 (64 symbols). -----
    vk_ep!("neg_f32",  crate::vulkan_dispatch::unary::neg_f32),
    vk_ep!("neg_f16",  crate::vulkan_dispatch::unary_f16::neg_f16),
    vk_ep!("neg_f64",  crate::vulkan_dispatch::unary_f64::neg_f64),
    vk_ep!("neg_bf16", crate::vulkan_dispatch::unary_bf16::neg_bf16),
    vk_ep!("sqr_f32",  crate::vulkan_dispatch::unary::sqr_f32),
    vk_ep!("sqr_f16",  crate::vulkan_dispatch::unary_f16::sqr_f16),
    vk_ep!("sqr_f64",  crate::vulkan_dispatch::unary_f64::sqr_f64),
    vk_ep!("sqr_bf16", crate::vulkan_dispatch::unary_bf16::sqr_bf16),
    vk_ep!("sqrt_f32",  crate::vulkan_dispatch::unary::sqrt_f32),
    vk_ep!("sqrt_f16",  crate::vulkan_dispatch::unary_f16::sqrt_f16),
    vk_ep!("sqrt_f64",  crate::vulkan_dispatch::unary_f64::sqrt_f64),
    vk_ep!("sqrt_bf16", crate::vulkan_dispatch::unary_bf16::sqrt_bf16),
    vk_ep!("exp_f32",  crate::vulkan_dispatch::unary::exp_f32),
    vk_ep!("exp_f16",  crate::vulkan_dispatch::unary_f16::exp_f16),
    vk_ep!("exp_f64",  crate::vulkan_dispatch::unary_f64::exp_f64),
    vk_ep!("exp_bf16", crate::vulkan_dispatch::unary_bf16::exp_bf16),
    vk_ep!("log_f32",  crate::vulkan_dispatch::unary::log_f32),
    vk_ep!("log_f16",  crate::vulkan_dispatch::unary_f16::log_f16),
    vk_ep!("log_f64",  crate::vulkan_dispatch::unary_f64::log_f64),
    vk_ep!("log_bf16", crate::vulkan_dispatch::unary_bf16::log_bf16),
    vk_ep!("sin_f32",  crate::vulkan_dispatch::unary::sin_f32),
    vk_ep!("sin_f16",  crate::vulkan_dispatch::unary_f16::sin_f16),
    vk_ep!("sin_f64",  crate::vulkan_dispatch::unary_f64::sin_f64),
    vk_ep!("sin_bf16", crate::vulkan_dispatch::unary_bf16::sin_bf16),
    vk_ep!("cos_f32",  crate::vulkan_dispatch::unary::cos_f32),
    vk_ep!("cos_f16",  crate::vulkan_dispatch::unary_f16::cos_f16),
    vk_ep!("cos_f64",  crate::vulkan_dispatch::unary_f64::cos_f64),
    vk_ep!("cos_bf16", crate::vulkan_dispatch::unary_bf16::cos_bf16),
    vk_ep!("tanh_f32",  crate::vulkan_dispatch::unary::tanh_f32),
    vk_ep!("tanh_f16",  crate::vulkan_dispatch::unary_f16::tanh_f16),
    vk_ep!("tanh_f64",  crate::vulkan_dispatch::unary_f64::tanh_f64),
    vk_ep!("tanh_bf16", crate::vulkan_dispatch::unary_bf16::tanh_bf16),
    vk_ep!("sigmoid_f32",  crate::vulkan_dispatch::unary::sigmoid_f32),
    vk_ep!("sigmoid_f16",  crate::vulkan_dispatch::unary_f16::sigmoid_f16),
    vk_ep!("sigmoid_f64",  crate::vulkan_dispatch::unary_f64::sigmoid_f64),
    vk_ep!("sigmoid_bf16", crate::vulkan_dispatch::unary_bf16::sigmoid_bf16),
    vk_ep!("silu_f32",  crate::vulkan_dispatch::unary::silu_f32),
    vk_ep!("silu_f16",  crate::vulkan_dispatch::unary_f16::silu_f16),
    vk_ep!("silu_f64",  crate::vulkan_dispatch::unary_f64::silu_f64),
    vk_ep!("silu_bf16", crate::vulkan_dispatch::unary_bf16::silu_bf16),
    vk_ep!("gelu_f32",  crate::vulkan_dispatch::unary::gelu_f32),
    vk_ep!("gelu_f16",  crate::vulkan_dispatch::unary_f16::gelu_f16),
    vk_ep!("gelu_f64",  crate::vulkan_dispatch::unary_f64::gelu_f64),
    vk_ep!("gelu_bf16", crate::vulkan_dispatch::unary_bf16::gelu_bf16),
    vk_ep!("relu_f32",  crate::vulkan_dispatch::unary::relu_f32),
    vk_ep!("relu_f16",  crate::vulkan_dispatch::unary_f16::relu_f16),
    vk_ep!("relu_f64",  crate::vulkan_dispatch::unary_f64::relu_f64),
    vk_ep!("relu_bf16", crate::vulkan_dispatch::unary_bf16::relu_bf16),
    vk_ep!("step_f32",  crate::vulkan_dispatch::unary::step_f32),
    vk_ep!("step_f16",  crate::vulkan_dispatch::unary_f16::step_f16),
    vk_ep!("step_f64",  crate::vulkan_dispatch::unary_f64::step_f64),
    vk_ep!("step_bf16", crate::vulkan_dispatch::unary_bf16::step_bf16),
    vk_ep!("abs_f32",  crate::vulkan_dispatch::unary::abs_f32),
    vk_ep!("abs_f16",  crate::vulkan_dispatch::unary_f16::abs_f16),
    vk_ep!("abs_f64",  crate::vulkan_dispatch::unary_f64::abs_f64),
    vk_ep!("abs_bf16", crate::vulkan_dispatch::unary_bf16::abs_bf16),
    vk_ep!("sign_f32",  crate::vulkan_dispatch::unary::sign_f32),
    vk_ep!("sign_f16",  crate::vulkan_dispatch::unary_f16::sign_f16),
    vk_ep!("sign_f64",  crate::vulkan_dispatch::unary_f64::sign_f64),
    vk_ep!("sign_bf16", crate::vulkan_dispatch::unary_bf16::sign_bf16),
    vk_ep!("recip_f32",  crate::vulkan_dispatch::unary::recip_f32),
    vk_ep!("recip_f16",  crate::vulkan_dispatch::unary_f16::recip_f16),
    vk_ep!("recip_f64",  crate::vulkan_dispatch::unary_f64::recip_f64),
    vk_ep!("recip_bf16", crate::vulkan_dispatch::unary_bf16::recip_bf16),
    // ----- Binary: 6 ops, strided f32/f16/f64/bf16 fan (24 symbols). -----
    vk_ep!("add_f32",  crate::vulkan_dispatch::binary::add_f32),
    vk_ep!("add_f16",  crate::vulkan_dispatch::binary_f16::add_f16),
    vk_ep!("add_f64",  crate::vulkan_dispatch::binary_f64::add_f64),
    vk_ep!("add_bf16", crate::vulkan_dispatch::binary_bf16::add_bf16),
    vk_ep!("sub_f32",  crate::vulkan_dispatch::binary::sub_f32),
    vk_ep!("sub_f16",  crate::vulkan_dispatch::binary_f16::sub_f16),
    vk_ep!("sub_f64",  crate::vulkan_dispatch::binary_f64::sub_f64),
    vk_ep!("sub_bf16", crate::vulkan_dispatch::binary_bf16::sub_bf16),
    vk_ep!("mul_f32",  crate::vulkan_dispatch::binary::mul_f32),
    vk_ep!("mul_f16",  crate::vulkan_dispatch::binary_f16::mul_f16),
    vk_ep!("mul_f64",  crate::vulkan_dispatch::binary_f64::mul_f64),
    vk_ep!("mul_bf16", crate::vulkan_dispatch::binary_bf16::mul_bf16),
    vk_ep!("div_f32",  crate::vulkan_dispatch::binary::div_f32),
    vk_ep!("div_f16",  crate::vulkan_dispatch::binary_f16::div_f16),
    vk_ep!("div_f64",  crate::vulkan_dispatch::binary_f64::div_f64),
    vk_ep!("div_bf16", crate::vulkan_dispatch::binary_bf16::div_bf16),
    vk_ep!("maximum_f32",  crate::vulkan_dispatch::binary::maximum_f32),
    vk_ep!("maximum_f16",  crate::vulkan_dispatch::binary_f16::maximum_f16),
    vk_ep!("maximum_f64",  crate::vulkan_dispatch::binary_f64::maximum_f64),
    vk_ep!("maximum_bf16", crate::vulkan_dispatch::binary_bf16::maximum_bf16),
    vk_ep!("minimum_f32",  crate::vulkan_dispatch::binary::minimum_f32),
    vk_ep!("minimum_f16",  crate::vulkan_dispatch::binary_f16::minimum_f16),
    vk_ep!("minimum_f64",  crate::vulkan_dispatch::binary_f64::minimum_f64),
    vk_ep!("minimum_bf16", crate::vulkan_dispatch::binary_bf16::minimum_bf16),
    // ----- Affine: strided f32/f16/f64 fan + contiguous bf16 (all in the `affine` module). -----
    vk_ep!("affine_f32", crate::vulkan_dispatch::affine::affine_f32),
    vk_ep!("affine_f16", crate::vulkan_dispatch::affine::affine_f16),
    vk_ep!("affine_f64", crate::vulkan_dispatch::affine::affine_f64),
    vk_ep!("affine_bf16", crate::vulkan_dispatch::affine::affine_bf16),
    // ----- Clamp / PowI: single-dtype f32. -----
    vk_ep!("clamp_f32", crate::vulkan_dispatch::clamp::clamp_f32),
    vk_ep!("powi_f32",  crate::vulkan_dispatch::powi::powi_f32),
];

/// The Vulkan matmul family's `symbol → production wrapper` map — the FULL
/// family: 6 per-combo `(MatMul, [lhs,rhs,out])` wrapper bindings. Contract:
/// `docs/kernel-contracts/vulkan/matmul.fkc.md`, re-authored per-combo (the cast
/// family's per-pair precedent). Each section declares a single dtype per operand
/// + a `fixed(OUT)` / `passthrough(lhs)` output, so none of them dtype-fan — the
/// importer keys `[lhs, rhs, out]` (inputs-then-output), byte-for-byte the deleted
/// hand-written `table.register_with_precision(OpKind::MatMul, &[lhs, rhs, out], …)`
/// regs.
///
/// Each symbol is one production wrapper that route-picks its internal Slang
/// kernels (matvec / reg-tile / tiled for f32; matvec_bf16_b / matmul_tiled_bf16_b
/// / matmul_coop* + matmul_small_* fallbacks for the mixed combos) at dispatch
/// time — the route-pick is NOT a table binding, so there is exactly one
/// `KernelRef` per key (no duplicate-at-key which `finalize` would reject). Caps
/// ride through contiguous-only (`awkward_layout_strategy: requires_contiguous`
/// ⇒ `strided_input == false`), byte-for-byte the deleted `register_with_precision`
/// regs (the coop / vec4 loads require canonical row-major; a strided operand is
/// auto-Contiguized first).
pub static VULKAN_MATMUL_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // f32 GEMM/GEMV wrapper (matvec / reg-tile / tiled route-pick).
    vk_ep!("matmul_f32",           crate::vulkan_dispatch::matmul::matmul_f32),
    // Mixed-precision f32 × bf16 → f32 (matvec_bf16_b / matmul_tiled_bf16_b / coop).
    vk_ep!("matmul_f32_bf16_b",    crate::vulkan_dispatch::matmul::matmul_f32_bf16_b),
    // Cooperative-matrix bf16 × bf16 → f32 (coop + matmul_small_bf16_bf16_f32).
    vk_ep!("matmul_bf16_bf16_f32", crate::vulkan_dispatch::matmul::matmul_bf16_bf16_f32),
    // Cooperative-matrix bf16 × bf16 → bf16, downcast store (coop + small).
    vk_ep!("matmul_bf16_bf16_bf16", crate::vulkan_dispatch::matmul::matmul_bf16_bf16_bf16),
    // Cooperative-matrix f16 × f16 → f16, downcast store (coop + small).
    vk_ep!("matmul_f16_f16_f16",   crate::vulkan_dispatch::matmul::matmul_f16_f16_f16),
    // Cooperative-matrix f16 × f16 → f32 (coop + matmul_small_f16_f16_f32).
    vk_ep!("matmul_f16_f16_f32",   crate::vulkan_dispatch::matmul::matmul_f16_f16_f32),
];

/// The Vulkan conv2d family's `symbol → production wrapper` map — the FULL
/// family: 3 per-(op, dtype) `(Conv2D, [x, weight, out])` wrapper bindings.
/// Contract: `docs/kernel-contracts/vulkan/conv.fkc.md`, re-authored per-(op,
/// dtype) (the matmul family's per-combo precedent). Each section declares a
/// single dtype per operand + a `passthrough(x)` output, so none of them
/// dtype-fan — the importer keys `[x, weight, out]` (inputs-then-output),
/// byte-for-byte the deleted hand-written
/// `table.register_with_precision(OpKind::Conv2D, &[T, T, T], …)` regs.
///
/// Each symbol is one production wrapper (`conv2d::conv2d_f32/bf16/f16`) that
/// runs the WHOLE conv internally (`VulkanBackend::conv2d_*_bytes`: NCHW im2col
/// → GEMM, the coop tensor-core GEMM for bf16/f16) — the two-stage pipeline is
/// NOT a table binding, so there is exactly one `KernelRef` per key. NO bias key
/// is mapped: the wrappers bail on a 3-input (bias) call, so the contract
/// declares no optional bias operand and each section keys only `[x, weight,
/// out]` (unlike the CPU conv family's optional-bias dual key). Caps ride
/// through contiguous-only (`requires_contiguous` ⇒ `strided_input == false`) —
/// conv2d_im2col reads canonical row-major NCHW, so a strided operand is
/// auto-Contiguized first.
pub static VULKAN_CONV_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // f32 conv (im2col + f32 GEMM route-pick).
    vk_ep!("conv2d_f32",  crate::vulkan_dispatch::conv2d::conv2d_f32),
    // bf16 conv (im2col_bf16 + cooperative-matrix bf16 GEMM).
    vk_ep!("conv2d_bf16", crate::vulkan_dispatch::conv2d::conv2d_bf16),
    // f16 conv (im2col + cooperative-matrix f16 GEMM).
    vk_ep!("conv2d_f16",  crate::vulkan_dispatch::conv2d::conv2d_f16),
];

/// The Vulkan **select** family's `symbol → production wrapper` map — the FULL
/// family: 16 (op, dtype) keys across IndexSelect (4), Gather (6), MaskedFill
/// (6). Contract: `docs/kernel-contracts/vulkan/select.fkc.md`, authored per-op
/// (the conv/matmul per-combo precedent, with dtype fan-out §3.4). Each section
/// declares a BASE `entry_point` fanned over its dtype list, so the importer
/// resolves `<base>_<suffix>` and keys `[data, index, out]` (data dtype + fixed
/// index dtype + passthrough(data) output), byte-for-byte the deleted hand-written
/// `register_with_precision(OpKind::{IndexSelect,Gather,MaskedFill}, …)` regs.
///
/// IndexSelect fans to FOUR distinct per-dtype wrappers
/// (`indexing::index_select_{f32,f16,bf16,f64}`). Gather + MaskedFill each dispatch
/// through ONE dtype-agnostic wrapper (`gather::gather`, `masked_fill::masked_fill`)
/// that picks its element byte-width from the output dtype — so every fanned dtype
/// symbol maps to the SAME `KernelRef` (a synthetic-base umbrella, the pad_cpu
/// precedent; distinct dtype keys ⇒ legal sibling registrations of one wrapper).
/// Caps ride through contiguous-only (`requires_contiguous` ⇒ `strided_input ==
/// false`) — byte-level data movers read/write own-shape flat buffers; a strided
/// operand is auto-Contiguized first.
pub static VULKAN_SELECT_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // IndexSelect: 4 distinct per-dtype wrappers.
    vk_ep!("index_select_f32",  crate::vulkan_dispatch::indexing::index_select_f32),
    vk_ep!("index_select_f16",  crate::vulkan_dispatch::indexing::index_select_f16),
    vk_ep!("index_select_bf16", crate::vulkan_dispatch::indexing::index_select_bf16),
    vk_ep!("index_select_f64",  crate::vulkan_dispatch::indexing::index_select_f64),
    // Gather: 6 dtype symbols → the ONE dtype-agnostic gather wrapper.
    vk_ep!("gather_f32",  crate::vulkan_dispatch::gather::gather),
    vk_ep!("gather_f16",  crate::vulkan_dispatch::gather::gather),
    vk_ep!("gather_bf16", crate::vulkan_dispatch::gather::gather),
    vk_ep!("gather_f64",  crate::vulkan_dispatch::gather::gather),
    vk_ep!("gather_u8",   crate::vulkan_dispatch::gather::gather),
    vk_ep!("gather_u32",  crate::vulkan_dispatch::gather::gather),
    // MaskedFill: 6 dtype symbols → the ONE dtype-agnostic masked_fill wrapper.
    vk_ep!("masked_fill_f32",  crate::vulkan_dispatch::masked_fill::masked_fill),
    vk_ep!("masked_fill_f16",  crate::vulkan_dispatch::masked_fill::masked_fill),
    vk_ep!("masked_fill_bf16", crate::vulkan_dispatch::masked_fill::masked_fill),
    vk_ep!("masked_fill_f64",  crate::vulkan_dispatch::masked_fill::masked_fill),
    vk_ep!("masked_fill_u8",   crate::vulkan_dispatch::masked_fill::masked_fill),
    vk_ep!("masked_fill_u32",  crate::vulkan_dispatch::masked_fill::masked_fill),
];

/// The Vulkan **scatter** family's `symbol → production wrapper` map — the FULL
/// family: 8 (op, dtype) keys across IndexAdd (4) + ScatterAdd (4). Contract:
/// `docs/kernel-contracts/vulkan/scatter.fkc.md`, authored per-op with dtype
/// fan-out (§3.4). Each section declares a BASE `entry_point` fanned over
/// `[F32, F64, BF16, F16]` (the `base` + `src` operands SHARE that list ⇒ they
/// fan together), so the importer resolves `<base>_<suffix>` and keys the 4-slot
/// `[base, U32, src, out]` (`passthrough(base)` output), byte-for-byte the deleted
/// hand-written `register_with_precision(OpKind::{IndexAdd,ScatterAdd}, …)` regs.
///
/// Each fanned symbol resolves to its distinct per-dtype wrapper
/// (`index_add::index_add_*`, `scatter_add::scatter_add_*`). Caps ride through
/// contiguous-only (`requires_contiguous` ⇒ `strided_input == false`); precision is
/// the contract's audited nondeterministic `none(reason)` seed (bounded-CAS atomic
/// accumulate, scheduler-dependent order).
pub static VULKAN_SCATTER_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // IndexAdd: 4 distinct per-dtype wrappers.
    vk_ep!("index_add_f32",  crate::vulkan_dispatch::index_add::index_add_f32),
    vk_ep!("index_add_f64",  crate::vulkan_dispatch::index_add::index_add_f64),
    vk_ep!("index_add_bf16", crate::vulkan_dispatch::index_add::index_add_bf16),
    vk_ep!("index_add_f16",  crate::vulkan_dispatch::index_add::index_add_f16),
    // ScatterAdd: 4 distinct per-dtype wrappers.
    vk_ep!("scatter_add_f32",  crate::vulkan_dispatch::scatter_add::scatter_add_f32),
    vk_ep!("scatter_add_f64",  crate::vulkan_dispatch::scatter_add::scatter_add_f64),
    vk_ep!("scatter_add_bf16", crate::vulkan_dispatch::scatter_add::scatter_add_bf16),
    vk_ep!("scatter_add_f16",  crate::vulkan_dispatch::scatter_add::scatter_add_f16),
];

/// The Vulkan **movement** family's `symbol → production wrapper` map — the FULL
/// family: 8 (op, dtype) keys across Concat (4) + CumSum (4). Contract:
/// `docs/kernel-contracts/vulkan/movement.fkc.md`, authored per-op with dtype
/// fan-out (§3.4). Each section declares a BASE `entry_point` fanned over its dtype
/// list, resolving `<base>_<suffix>` to each distinct per-dtype wrapper and keying
/// `[T, T]` (`passthrough(input)` output), byte-for-byte the deleted hand-written
/// `register_with_caps_and_precision(OpKind::{Concat,CumSum}, …, strided, …)` regs.
///
/// Both are STRIDE-AWARE (`strided: accepted` ⇒ `strided_input == true`) — the
/// caps-through-import proof for this family. Concat precision is byte-exact
/// (bitwise); CumSum takes the conservative UNAUDITED author seed (the elementwise
/// pointwise precedent).
pub static VULKAN_MOVEMENT_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // Concat: 4 distinct per-dtype wrappers.
    vk_ep!("concat_f32",  crate::vulkan_dispatch::concat::concat_f32),
    vk_ep!("concat_f16",  crate::vulkan_dispatch::concat::concat_f16),
    vk_ep!("concat_bf16", crate::vulkan_dispatch::concat::concat_bf16),
    vk_ep!("concat_f64",  crate::vulkan_dispatch::concat::concat_f64),
    // CumSum: 4 distinct per-dtype wrappers.
    vk_ep!("cumsum_f32",  crate::vulkan_dispatch::cumsum::cumsum_f32),
    vk_ep!("cumsum_f64",  crate::vulkan_dispatch::cumsum::cumsum_f64),
    vk_ep!("cumsum_f16",  crate::vulkan_dispatch::cumsum::cumsum_f16),
    vk_ep!("cumsum_bf16", crate::vulkan_dispatch::cumsum::cumsum_bf16),
];

/// The Vulkan **shape** family's `symbol → production wrapper` map — the FULL
/// family: 28 (op, dtype) keys across Triu (7) + Tril (7) + Flip (7) + Roll (7)
/// over `[F32, F16, BF16, F64, I32, U32, I64]`. Contract:
/// `docs/kernel-contracts/vulkan/shape.fkc.md`, authored per-op with dtype fan-out
/// (§3.4). Each op is ONE dtype-agnostic wrapper (`triangular::triu` /
/// `triangular::tril` / `flip::flip` / `roll::roll`) that picks its byte-width from
/// the dtype — so every fanned `<base>_<suffix>` symbol maps to the SAME `KernelRef`
/// (a synthetic-base umbrella, the pad_cpu precedent). Distinct dtype keys ⇒ legal
/// sibling registrations of one wrapper, byte-for-byte the deleted hand-written
/// regs.
///
/// Caps split by op: Triu/Tril `requires_contiguous` (`strided_input == false`),
/// Flip/Roll `strided: accepted` (`strided_input == true`) — the caps-through-import
/// proof. All are byte-exact (bitwise, `max_ulp: 0`).
pub static VULKAN_SHAPE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // Triu: 7 dtype symbols → the ONE dtype-agnostic triu wrapper.
    vk_ep!("triu_f32",  crate::vulkan_dispatch::triangular::triu),
    vk_ep!("triu_f16",  crate::vulkan_dispatch::triangular::triu),
    vk_ep!("triu_bf16", crate::vulkan_dispatch::triangular::triu),
    vk_ep!("triu_f64",  crate::vulkan_dispatch::triangular::triu),
    vk_ep!("triu_i32",  crate::vulkan_dispatch::triangular::triu),
    vk_ep!("triu_u32",  crate::vulkan_dispatch::triangular::triu),
    vk_ep!("triu_i64",  crate::vulkan_dispatch::triangular::triu),
    // Tril: 7 dtype symbols → the ONE dtype-agnostic tril wrapper.
    vk_ep!("tril_f32",  crate::vulkan_dispatch::triangular::tril),
    vk_ep!("tril_f16",  crate::vulkan_dispatch::triangular::tril),
    vk_ep!("tril_bf16", crate::vulkan_dispatch::triangular::tril),
    vk_ep!("tril_f64",  crate::vulkan_dispatch::triangular::tril),
    vk_ep!("tril_i32",  crate::vulkan_dispatch::triangular::tril),
    vk_ep!("tril_u32",  crate::vulkan_dispatch::triangular::tril),
    vk_ep!("tril_i64",  crate::vulkan_dispatch::triangular::tril),
    // Flip: 7 dtype symbols → the ONE dtype-agnostic flip wrapper (stride-aware).
    vk_ep!("flip_f32",  crate::vulkan_dispatch::flip::flip),
    vk_ep!("flip_f16",  crate::vulkan_dispatch::flip::flip),
    vk_ep!("flip_bf16", crate::vulkan_dispatch::flip::flip),
    vk_ep!("flip_f64",  crate::vulkan_dispatch::flip::flip),
    vk_ep!("flip_i32",  crate::vulkan_dispatch::flip::flip),
    vk_ep!("flip_u32",  crate::vulkan_dispatch::flip::flip),
    vk_ep!("flip_i64",  crate::vulkan_dispatch::flip::flip),
    // Roll: 7 dtype symbols → the ONE dtype-agnostic roll wrapper (stride-aware).
    vk_ep!("roll_f32",  crate::vulkan_dispatch::roll::roll),
    vk_ep!("roll_f16",  crate::vulkan_dispatch::roll::roll),
    vk_ep!("roll_bf16", crate::vulkan_dispatch::roll::roll),
    vk_ep!("roll_f64",  crate::vulkan_dispatch::roll::roll),
    vk_ep!("roll_i32",  crate::vulkan_dispatch::roll::roll),
    vk_ep!("roll_u32",  crate::vulkan_dispatch::roll::roll),
    vk_ep!("roll_i64",  crate::vulkan_dispatch::roll::roll),
];

/// The Vulkan **pad-copy** family's `symbol → production wrapper` map — the FULL
/// family: 21 (op, dtype) keys across Pad (6) + PadBackward (6) + Copy (9).
/// Contract: `docs/kernel-contracts/vulkan/pad-copy.fkc.md`, authored per-op with
/// dtype fan-out (§3.4). Each op is ONE dtype-agnostic wrapper (`pad::pad_const` /
/// `pad::pad_backward` / `copy_to_cpu_vulkan`) that picks its byte-width from the
/// dtype — so every fanned `<base>_<suffix>` symbol maps to the SAME `KernelRef`
/// (a synthetic-base umbrella, the pad_cpu precedent). Distinct dtype keys ⇒ legal
/// sibling registrations of one wrapper, byte-for-byte the deleted hand-written
/// regs. Caps ride through contiguous-only (`requires_contiguous` ⇒
/// `strided_input == false`); precision byte-exact (bitwise, `max_ulp: 0`).
pub static VULKAN_PAD_COPY_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // Pad: 6 dtype symbols → the ONE dtype-agnostic pad_const wrapper.
    vk_ep!("pad_f32",  crate::vulkan_dispatch::pad::pad_const),
    vk_ep!("pad_f16",  crate::vulkan_dispatch::pad::pad_const),
    vk_ep!("pad_bf16", crate::vulkan_dispatch::pad::pad_const),
    vk_ep!("pad_f64",  crate::vulkan_dispatch::pad::pad_const),
    vk_ep!("pad_u8",   crate::vulkan_dispatch::pad::pad_const),
    vk_ep!("pad_u32",  crate::vulkan_dispatch::pad::pad_const),
    // PadBackward: 6 dtype symbols → the ONE dtype-agnostic pad_backward wrapper.
    vk_ep!("pad_backward_f32",  crate::vulkan_dispatch::pad::pad_backward),
    vk_ep!("pad_backward_f16",  crate::vulkan_dispatch::pad::pad_backward),
    vk_ep!("pad_backward_bf16", crate::vulkan_dispatch::pad::pad_backward),
    vk_ep!("pad_backward_f64",  crate::vulkan_dispatch::pad::pad_backward),
    vk_ep!("pad_backward_u8",   crate::vulkan_dispatch::pad::pad_backward),
    vk_ep!("pad_backward_u32",  crate::vulkan_dispatch::pad::pad_backward),
    // Copy: 9 dtype symbols → the ONE dtype-agnostic D2H copy wrapper.
    vk_ep!("copy_f32",  crate::vulkan_dispatch::copy_to_cpu_vulkan),
    vk_ep!("copy_f16",  crate::vulkan_dispatch::copy_to_cpu_vulkan),
    vk_ep!("copy_bf16", crate::vulkan_dispatch::copy_to_cpu_vulkan),
    vk_ep!("copy_f64",  crate::vulkan_dispatch::copy_to_cpu_vulkan),
    vk_ep!("copy_u32",  crate::vulkan_dispatch::copy_to_cpu_vulkan),
    vk_ep!("copy_u8",   crate::vulkan_dispatch::copy_to_cpu_vulkan),
    vk_ep!("copy_i16",  crate::vulkan_dispatch::copy_to_cpu_vulkan),
    vk_ep!("copy_i32",  crate::vulkan_dispatch::copy_to_cpu_vulkan),
    vk_ep!("copy_i64",  crate::vulkan_dispatch::copy_to_cpu_vulkan),
];

/// The Vulkan **write-slice** family's `symbol → production wrapper` map — the
/// FULL family: 18 (op, dtype) keys across WriteSlice (9) + WriteSliceRotating (9).
/// Contract: `docs/kernel-contracts/vulkan/write-slice.fkc.md`, authored per-op
/// with dtype fan-out (§3.4). BYTE-WIDTH-keyed: each op's 9 dtype symbols collapse
/// to FOUR wrappers by element size (`b1/b2/b4/b8`) — the cast family's "several
/// sections share one wrapper" precedent. The link registry maps each fanned
/// `<base>_<suffix>` symbol to the byte-width wrapper for that dtype's size, keying
/// `[T, T]` byte-for-byte the deleted hand-written
/// `register_with_precision(OpKind::{WriteSlice,WriteSliceRotating}, …)` regs. Caps
/// ride through contiguous-only (`requires_contiguous` ⇒ `strided_input == false`);
/// precision byte-exact (bitwise, `max_ulp: 0`).
pub static VULKAN_WRITE_SLICE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // WriteSlice: 9 dtype symbols → 4 byte-width wrappers (b4/b2/b8/b1).
    vk_ep!("write_slice_f32",  crate::vulkan_dispatch::write_slice::write_slice_b4),
    vk_ep!("write_slice_i32",  crate::vulkan_dispatch::write_slice::write_slice_b4),
    vk_ep!("write_slice_u32",  crate::vulkan_dispatch::write_slice::write_slice_b4),
    vk_ep!("write_slice_f16",  crate::vulkan_dispatch::write_slice::write_slice_b2),
    vk_ep!("write_slice_bf16", crate::vulkan_dispatch::write_slice::write_slice_b2),
    vk_ep!("write_slice_f64",  crate::vulkan_dispatch::write_slice::write_slice_b8),
    vk_ep!("write_slice_i64",  crate::vulkan_dispatch::write_slice::write_slice_b8),
    vk_ep!("write_slice_u8",   crate::vulkan_dispatch::write_slice::write_slice_b1),
    vk_ep!("write_slice_i8",   crate::vulkan_dispatch::write_slice::write_slice_b1),
    // WriteSliceRotating: 9 dtype symbols → 4 byte-width rotating wrappers.
    vk_ep!("write_slice_rotating_f32",  crate::vulkan_dispatch::write_slice_rotating::write_slice_rotating_b4),
    vk_ep!("write_slice_rotating_i32",  crate::vulkan_dispatch::write_slice_rotating::write_slice_rotating_b4),
    vk_ep!("write_slice_rotating_u32",  crate::vulkan_dispatch::write_slice_rotating::write_slice_rotating_b4),
    vk_ep!("write_slice_rotating_f16",  crate::vulkan_dispatch::write_slice_rotating::write_slice_rotating_b2),
    vk_ep!("write_slice_rotating_bf16", crate::vulkan_dispatch::write_slice_rotating::write_slice_rotating_b2),
    vk_ep!("write_slice_rotating_f64",  crate::vulkan_dispatch::write_slice_rotating::write_slice_rotating_b8),
    vk_ep!("write_slice_rotating_i64",  crate::vulkan_dispatch::write_slice_rotating::write_slice_rotating_b8),
    vk_ep!("write_slice_rotating_u8",   crate::vulkan_dispatch::write_slice_rotating::write_slice_rotating_b1),
    vk_ep!("write_slice_rotating_i8",   crate::vulkan_dispatch::write_slice_rotating::write_slice_rotating_b1),
];

/// The Vulkan **rope** family's `symbol → production wrapper` map — the FULL
/// family: 4 (Rope, [x, cos, sin, out]) per-dtype bindings. Contract:
/// `docs/kernel-contracts/vulkan/rope.fkc.md`, authored with dtype fan-out (§3.4):
/// the single section declares a BASE `entry_point` fanned over `[F32, F16, F64,
/// BF16]` (the `x` + `cos` + `sin` operands SHARE that list ⇒ they fan together),
/// resolving `rope_<suffix>` to each distinct per-dtype wrapper and keying the
/// 4-slot `[x, cos, sin, out]` (`passthrough(x)` output), byte-for-byte the deleted
/// hand-written `register_with_caps_and_precision(OpKind::Rope, …, strided, …)`
/// regs. STRIDE-AWARE (`strided: accepted` ⇒ `strided_input == true`) — the
/// caps-through-import proof; precision is the conservative UNAUDITED author seed
/// (the elementwise pointwise precedent).
pub static VULKAN_ROPE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    vk_ep!("rope_f32",  crate::vulkan_dispatch::attention::rope_f32),
    vk_ep!("rope_f16",  crate::vulkan_dispatch::attention::rope_f16),
    vk_ep!("rope_f64",  crate::vulkan_dispatch::attention::rope_f64),
    vk_ep!("rope_bf16", crate::vulkan_dispatch::attention::rope_bf16),
];

/// The Vulkan **reduce** family's `symbol → production wrapper` map — the FULL
/// family: 24 (op, dtype) keys across SumReduce / MaxReduce / MinReduce /
/// MeanReduce (16, key `[T, T]`) + ArgMaxDim / ArgMinDim (8, key `[T, U32]`).
/// Contract: `docs/kernel-contracts/vulkan/reduce-prims.fkc.md`, authored per-op
/// with dtype fan-out (§3.4) — the production per-(op, dtype) binding model that
/// supersedes the aspirational `vulkan/reduce.fkc.md` op-id-selector corpus. Each
/// section fans a BASE `entry_point` over `[F32, F16, BF16, F64]` to the DISTINCT
/// per-dtype wrapper, byte-for-byte the deleted hand-written
/// `register_with_precision(OpKind::{SumReduce,…,ArgMinDim}, …)` regs. Caps ride
/// through contiguous-only (`requires_contiguous` ⇒ `strided_input == false`);
/// value reduces are audited nondeterministic none(reason), arg reduces bitwise
/// exact.
pub static VULKAN_REDUCE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // Value reduces: 4 ops × 4 dtypes → 16 distinct wrappers.
    vk_ep!("reduce_sum_f32",  crate::vulkan_dispatch::reduce::sum_f32),
    vk_ep!("reduce_sum_f16",  crate::vulkan_dispatch::reduce::sum_f16),
    vk_ep!("reduce_sum_bf16", crate::vulkan_dispatch::reduce::sum_bf16),
    vk_ep!("reduce_sum_f64",  crate::vulkan_dispatch::reduce::sum_f64),
    vk_ep!("reduce_max_f32",  crate::vulkan_dispatch::reduce::max_f32),
    vk_ep!("reduce_max_f16",  crate::vulkan_dispatch::reduce::max_f16),
    vk_ep!("reduce_max_bf16", crate::vulkan_dispatch::reduce::max_bf16),
    vk_ep!("reduce_max_f64",  crate::vulkan_dispatch::reduce::max_f64),
    vk_ep!("reduce_min_f32",  crate::vulkan_dispatch::reduce::min_f32),
    vk_ep!("reduce_min_f16",  crate::vulkan_dispatch::reduce::min_f16),
    vk_ep!("reduce_min_bf16", crate::vulkan_dispatch::reduce::min_bf16),
    vk_ep!("reduce_min_f64",  crate::vulkan_dispatch::reduce::min_f64),
    vk_ep!("reduce_mean_f32",  crate::vulkan_dispatch::reduce::mean_f32),
    vk_ep!("reduce_mean_f16",  crate::vulkan_dispatch::reduce::mean_f16),
    vk_ep!("reduce_mean_bf16", crate::vulkan_dispatch::reduce::mean_bf16),
    vk_ep!("reduce_mean_f64",  crate::vulkan_dispatch::reduce::mean_f64),
    // Index reduces: 2 ops × 4 dtypes → 8 distinct wrappers.
    vk_ep!("arg_max_f32",  crate::vulkan_dispatch::arg_reduce::argmax_f32),
    vk_ep!("arg_max_f16",  crate::vulkan_dispatch::arg_reduce::argmax_f16),
    vk_ep!("arg_max_bf16", crate::vulkan_dispatch::arg_reduce::argmax_bf16),
    vk_ep!("arg_max_f64",  crate::vulkan_dispatch::arg_reduce::argmax_f64),
    vk_ep!("arg_min_f32",  crate::vulkan_dispatch::arg_reduce::argmin_f32),
    vk_ep!("arg_min_f16",  crate::vulkan_dispatch::arg_reduce::argmin_f16),
    vk_ep!("arg_min_bf16", crate::vulkan_dispatch::arg_reduce::argmin_bf16),
    vk_ep!("arg_min_f64",  crate::vulkan_dispatch::arg_reduce::argmin_f64),
];

/// The Vulkan **norm** family's `symbol → production wrapper` map — the FULL
/// family: 20 (op, dtype) keys across SoftmaxLastDim (4) + SoftmaxLastDimBackward
/// (4) + LayerNormLastDim (4) + LayerNormLastDimBackward (4) + RmsNormLastDim (4).
/// Contract: `docs/kernel-contracts/vulkan/norm.fkc.md`, authored per-op with dtype
/// fan-out (§3.4). Each section fans a BASE `entry_point` over `[F32, F16, BF16,
/// F64]` to the DISTINCT per-dtype wrapper (`softmax::*`, `norm::*`), keying forward
/// families `[T, T]` and backward families `[T, T, T]` (`[y,g,dx]` / `[x,g,dx]`,
/// both inputs sharing the fanned list), byte-for-byte the deleted hand-written
/// `register_with_precision(OpKind::{SoftmaxLastDim,…,RmsNormLastDim}, …)` regs.
/// Caps ride through contiguous-only (`requires_contiguous` ⇒ `strided_input ==
/// false`); precision is the audited nondeterministic none(reason) seed
/// (per-row subgroup-tree FADD order).
pub static VULKAN_NORM_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // Softmax forward + backward.
    vk_ep!("softmax_last_dim_f32",  crate::vulkan_dispatch::softmax::softmax_f32),
    vk_ep!("softmax_last_dim_f16",  crate::vulkan_dispatch::softmax::softmax_f16),
    vk_ep!("softmax_last_dim_bf16", crate::vulkan_dispatch::softmax::softmax_bf16),
    vk_ep!("softmax_last_dim_f64",  crate::vulkan_dispatch::softmax::softmax_f64),
    vk_ep!("softmax_last_dim_backward_f32",  crate::vulkan_dispatch::softmax::softmax_last_dim_backward_f32),
    vk_ep!("softmax_last_dim_backward_f16",  crate::vulkan_dispatch::softmax::softmax_last_dim_backward_f16),
    vk_ep!("softmax_last_dim_backward_bf16", crate::vulkan_dispatch::softmax::softmax_last_dim_backward_bf16),
    vk_ep!("softmax_last_dim_backward_f64",  crate::vulkan_dispatch::softmax::softmax_last_dim_backward_f64),
    // LayerNorm forward + backward.
    vk_ep!("layer_norm_last_dim_f32",  crate::vulkan_dispatch::norm::layer_norm_f32),
    vk_ep!("layer_norm_last_dim_f16",  crate::vulkan_dispatch::norm::layer_norm_f16),
    vk_ep!("layer_norm_last_dim_bf16", crate::vulkan_dispatch::norm::layer_norm_bf16),
    vk_ep!("layer_norm_last_dim_f64",  crate::vulkan_dispatch::norm::layer_norm_f64),
    vk_ep!("layer_norm_last_dim_backward_f32",  crate::vulkan_dispatch::norm::layer_norm_backward_f32),
    vk_ep!("layer_norm_last_dim_backward_f16",  crate::vulkan_dispatch::norm::layer_norm_backward_f16),
    vk_ep!("layer_norm_last_dim_backward_bf16", crate::vulkan_dispatch::norm::layer_norm_backward_bf16),
    vk_ep!("layer_norm_last_dim_backward_f64",  crate::vulkan_dispatch::norm::layer_norm_backward_f64),
    // RmsNorm forward.
    vk_ep!("rms_norm_last_dim_f32",  crate::vulkan_dispatch::norm::rms_f32),
    vk_ep!("rms_norm_last_dim_f16",  crate::vulkan_dispatch::norm::rms_f16),
    vk_ep!("rms_norm_last_dim_bf16", crate::vulkan_dispatch::norm::rms_bf16),
    vk_ep!("rms_norm_last_dim_f64",  crate::vulkan_dispatch::norm::rms_f64),
];

/// The built-in Vulkan backend's [`LinkRegistry`] — resolves a contract's
/// `entry_point` symbols against [`VULKAN_CAST_ENTRY_POINTS`] +
/// [`VULKAN_ELEMENTWISE_ENTRY_POINTS`] + [`VULKAN_MATMUL_ENTRY_POINTS`] +
/// [`VULKAN_CONV_ENTRY_POINTS`] (and the future Vulkan families as they migrate
/// off their hand-written regs). Unresolved → `None`, which the importer turns
/// into a typed `UnknownEntryPoint` error (never a panic, never a fabricated
/// pointer).
pub struct VulkanLinkRegistry;

impl LinkRegistry for VulkanLinkRegistry {
    fn resolve_primitive(&self, symbol: &str) -> Option<KernelRef> {
        // Chain every migrated Vulkan family's table; each symbol is unique
        // across families (cast_*, per-op elementwise, matmul_*, conv2d_*), so
        // order is immaterial.
        VULKAN_CAST_ENTRY_POINTS
            .iter()
            .chain(VULKAN_ELEMENTWISE_ENTRY_POINTS.iter())
            .chain(VULKAN_MATMUL_ENTRY_POINTS.iter())
            .chain(VULKAN_CONV_ENTRY_POINTS.iter())
            .chain(VULKAN_SELECT_ENTRY_POINTS.iter())
            .chain(VULKAN_SCATTER_ENTRY_POINTS.iter())
            .chain(VULKAN_MOVEMENT_ENTRY_POINTS.iter())
            .chain(VULKAN_SHAPE_ENTRY_POINTS.iter())
            .chain(VULKAN_PAD_COPY_ENTRY_POINTS.iter())
            .chain(VULKAN_WRITE_SLICE_ENTRY_POINTS.iter())
            .chain(VULKAN_ROPE_ENTRY_POINTS.iter())
            .chain(VULKAN_REDUCE_ENTRY_POINTS.iter())
            .chain(VULKAN_NORM_ENTRY_POINTS.iter())
            .find(|(s, _)| *s == symbol)
            .map(|(_, k)| *k)
    }

    fn resolve_fused(&self, _symbol: &str) -> Option<KernelRef> {
        // No fused-op contracts in the Vulkan cast / elementwise corpus — every
        // section is a primitive `op_kind`.
        None
    }
}
