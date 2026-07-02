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

/// The CPU out-of-place scalar-param family's `symbol → production wrapper`
/// map (affine / clamp / powi × 4 dtypes + powi_backward × 4 = 16 kernels).
/// Contract: `docs/kernel-contracts/cpu/affine-clamp-powi.fkc.md`. The scalar
/// params (affine mul/add, clamp min/max, powi exp) ride in `OpParams`, NOT the
/// dtype-list, so the binding keys stay `[t, t]` for the single-input forward
/// ops and `[t, t, t]` for the two-input `powi_backward`. The `ep!` symbol is
/// built from `$op`/`$dt`, so the three f32 hand-written wrappers whose fn-name
/// differs from the symbol (`clamp_elementwise_f32`, `powi_elementwise_f32`)
/// still map to the contract's `clamp_f32` / `powi_f32` entry points.
pub static CPU_AFFINE_CLAMP_POWI_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // affine (y = mul*x + add)
    ep!("affine", "f32",  affine_f32_cpu_wrapper),
    ep!("affine", "f64",  affine_f64_cpu_wrapper),
    ep!("affine", "bf16", affine_bf16_cpu_wrapper),
    ep!("affine", "f16",  affine_f16_cpu_wrapper),
    // clamp (y = clamp(x, min, max))
    ep!("clamp",  "f32",  clamp_elementwise_f32_cpu_wrapper),
    ep!("clamp",  "f64",  clamp_f64_cpu_wrapper),
    ep!("clamp",  "bf16", clamp_bf16_cpu_wrapper),
    ep!("clamp",  "f16",  clamp_f16_cpu_wrapper),
    // powi (y = x.powi(exp))
    ep!("powi",   "f32",  powi_elementwise_f32_cpu_wrapper),
    ep!("powi",   "f64",  powi_f64_cpu_wrapper),
    ep!("powi",   "bf16", powi_bf16_cpu_wrapper),
    ep!("powi",   "f16",  powi_f16_cpu_wrapper),
    // powi_backward (grad_x = exp*x^(exp-1)*upstream) — TWO inputs (x, upstream)
    ep!("powi_backward", "f32",  powi_backward_f32_cpu_wrapper),
    ep!("powi_backward", "f64",  powi_backward_f64_cpu_wrapper),
    ep!("powi_backward", "bf16", powi_backward_bf16_cpu_wrapper),
    ep!("powi_backward", "f16",  powi_backward_f16_cpu_wrapper),
];

/// The built-in CPU backend's [`LinkRegistry`] — resolves a contract's
/// `entry_point` symbols against [`CPU_BINARY_ENTRY_POINTS`] and
/// [`CPU_AFFINE_CLAMP_POWI_ENTRY_POINTS`]. Unresolved → `None`, which the
/// importer turns into a typed `UnknownEntryPoint` error (never a panic, never
/// a fabricated pointer).
pub struct CpuLinkRegistry;

impl LinkRegistry for CpuLinkRegistry {
    fn resolve_primitive(&self, symbol: &str) -> Option<KernelRef> {
        CPU_BINARY_ENTRY_POINTS
            .iter()
            .chain(CPU_AFFINE_CLAMP_POWI_ENTRY_POINTS.iter())
            .find(|(s, _)| *s == symbol)
            .map(|(_, k)| *k)
    }

    fn resolve_fused(&self, _symbol: &str) -> Option<KernelRef> {
        // No fused-op contracts in the elementwise-binary or
        // affine/clamp/powi corpora.
        None
    }
}
