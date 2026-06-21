//! The built-in CPU backend's FKC `link_registry` (kernel-seam-interop §3.5,
//! §4.3; FKC §12.6). Maps each CPU kernel contract's `entry_point` symbol to
//! the production dispatch wrapper — the real, non-stub resolution the importer
//! uses so an imported contract binds the **actual** kernel (no raw pointers in
//! the serialized contract, FKC P9).
//!
//! For the built-in CPU backend the wrappers and this table co-locate in
//! fuel-dispatch — the dispatch layer that adapts raw byte-kernels to
//! [`KernelRef`]. An *external* provider (e.g. Baracuda) instead exports its own
//! link registry across the FFI; this is Fuel's internal-provider analogue, and
//! the first FKC conformance reference.

use crate::fkc::lower::LinkRegistry;
use crate::kernel::KernelRef;

/// One `(contract entry_point symbol, production wrapper)` pair. The symbol
/// matches the contract's `entry_point: "fuel_cpu_backend::byte_kernels::<op>_<dt>"`.
macro_rules! ep {
    ($op:literal, $dt:literal, $wrapper:ident) => {
        (
            concat!("fuel_cpu_backend::byte_kernels::", $op, "_", $dt),
            crate::dispatch::$wrapper as KernelRef,
        )
    };
}

/// The CPU elementwise-binary family's `symbol → production wrapper` map
/// (8 ops × 4 dtypes). The chassis umbrella section is `registrable: false`
/// (§3.10 describe-only), so it never reaches resolution and is absent here.
pub static CPU_BINARY_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    ep!("add", "f32", add_elementwise_f32_cpu_wrapper),
    ep!("add", "f64", add_elementwise_f64_cpu_wrapper),
    ep!("add", "f16", add_elementwise_f16_cpu_wrapper),
    ep!("add", "bf16", add_elementwise_bf16_cpu_wrapper),
    ep!("sub", "f32", sub_elementwise_f32_cpu_wrapper),
    ep!("sub", "f64", sub_elementwise_f64_cpu_wrapper),
    ep!("sub", "f16", sub_elementwise_f16_cpu_wrapper),
    ep!("sub", "bf16", sub_elementwise_bf16_cpu_wrapper),
    ep!("mul", "f32", mul_elementwise_f32_cpu_wrapper),
    ep!("mul", "f64", mul_elementwise_f64_cpu_wrapper),
    ep!("mul", "f16", mul_elementwise_f16_cpu_wrapper),
    ep!("mul", "bf16", mul_elementwise_bf16_cpu_wrapper),
    ep!("div", "f32", div_elementwise_f32_cpu_wrapper),
    ep!("div", "f64", div_elementwise_f64_cpu_wrapper),
    ep!("div", "f16", div_elementwise_f16_cpu_wrapper),
    ep!("div", "bf16", div_elementwise_bf16_cpu_wrapper),
    ep!("maximum", "f32", maximum_elementwise_f32_cpu_wrapper),
    ep!("maximum", "f64", maximum_elementwise_f64_cpu_wrapper),
    ep!("maximum", "f16", maximum_elementwise_f16_cpu_wrapper),
    ep!("maximum", "bf16", maximum_elementwise_bf16_cpu_wrapper),
    ep!("minimum", "f32", minimum_elementwise_f32_cpu_wrapper),
    ep!("minimum", "f64", minimum_elementwise_f64_cpu_wrapper),
    ep!("minimum", "f16", minimum_elementwise_f16_cpu_wrapper),
    ep!("minimum", "bf16", minimum_elementwise_bf16_cpu_wrapper),
    ep!("pow", "f32", pow_elementwise_f32_cpu_wrapper),
    ep!("pow", "f64", pow_elementwise_f64_cpu_wrapper),
    ep!("pow", "f16", pow_elementwise_f16_cpu_wrapper),
    ep!("pow", "bf16", pow_elementwise_bf16_cpu_wrapper),
    ep!("rem", "f32", rem_elementwise_f32_cpu_wrapper),
    ep!("rem", "f64", rem_elementwise_f64_cpu_wrapper),
    ep!("rem", "f16", rem_elementwise_f16_cpu_wrapper),
    ep!("rem", "bf16", rem_elementwise_bf16_cpu_wrapper),
];

/// The built-in CPU backend's [`LinkRegistry`] — resolves a contract's
/// `entry_point` symbols against [`CPU_BINARY_ENTRY_POINTS`]. Unresolved →
/// `None`, which the importer turns into a typed `UnknownEntryPoint` error
/// (never a panic, never a fabricated pointer).
pub struct CpuLinkRegistry;

impl LinkRegistry for CpuLinkRegistry {
    fn resolve_primitive(&self, symbol: &str) -> Option<KernelRef> {
        CPU_BINARY_ENTRY_POINTS
            .iter()
            .find(|(s, _)| *s == symbol)
            .map(|(_, k)| *k)
    }

    fn resolve_fused(&self, _symbol: &str) -> Option<KernelRef> {
        // No fused-op contracts in the elementwise-binary corpus.
        None
    }
}
