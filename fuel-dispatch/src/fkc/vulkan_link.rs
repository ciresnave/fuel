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

/// The built-in Vulkan backend's [`LinkRegistry`] — resolves a contract's
/// `entry_point` symbols against [`VULKAN_CAST_ENTRY_POINTS`] +
/// [`VULKAN_ELEMENTWISE_ENTRY_POINTS`] (and the future Vulkan families as they
/// migrate off their hand-written regs). Unresolved → `None`, which the importer
/// turns into a typed `UnknownEntryPoint` error (never a panic, never a
/// fabricated pointer).
pub struct VulkanLinkRegistry;

impl LinkRegistry for VulkanLinkRegistry {
    fn resolve_primitive(&self, symbol: &str) -> Option<KernelRef> {
        // Chain every migrated Vulkan family's table; each symbol is unique
        // across families (cast_*, per-op elementwise, matmul_*), so order is
        // immaterial.
        VULKAN_CAST_ENTRY_POINTS
            .iter()
            .chain(VULKAN_ELEMENTWISE_ENTRY_POINTS.iter())
            .chain(VULKAN_MATMUL_ENTRY_POINTS.iter())
            .find(|(s, _)| *s == symbol)
            .map(|(_, k)| *k)
    }

    fn resolve_fused(&self, _symbol: &str) -> Option<KernelRef> {
        // No fused-op contracts in the Vulkan cast / elementwise corpus — every
        // section is a primitive `op_kind`.
        None
    }
}
