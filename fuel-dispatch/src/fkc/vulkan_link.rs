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

/// The built-in Vulkan backend's [`LinkRegistry`] — resolves a contract's
/// `entry_point` symbols against [`VULKAN_CAST_ENTRY_POINTS`] (and the future
/// Vulkan families as they migrate off their hand-written regs). Unresolved →
/// `None`, which the importer turns into a typed `UnknownEntryPoint` error
/// (never a panic, never a fabricated pointer).
pub struct VulkanLinkRegistry;

impl LinkRegistry for VulkanLinkRegistry {
    fn resolve_primitive(&self, symbol: &str) -> Option<KernelRef> {
        VULKAN_CAST_ENTRY_POINTS
            .iter()
            .find(|(s, _)| *s == symbol)
            .map(|(_, k)| *k)
    }

    fn resolve_fused(&self, _symbol: &str) -> Option<KernelRef> {
        // No fused-op contracts in the Vulkan cast corpus — every section is an
        // `op_kind: Cast` primitive.
        None
    }
}
