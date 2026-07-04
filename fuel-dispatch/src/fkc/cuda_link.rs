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
use crate::kernel::KernelRef;

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

/// The built-in CUDA (baracuda) backend's [`LinkRegistry`] — resolves a
/// contract's `entry_point` symbols against [`CUDA_CAST_ENTRY_POINTS`] (and the
/// future CUDA families as they migrate off their hand-written regs). Unresolved
/// → `None`, which the importer turns into a typed `UnknownEntryPoint` error
/// (never a panic, never a fabricated pointer). Mirrors
/// [`crate::fkc::VulkanLinkRegistry`].
pub struct CudaLinkRegistry;

impl LinkRegistry for CudaLinkRegistry {
    fn resolve_primitive(&self, symbol: &str) -> Option<KernelRef> {
        // Chain every migrated CUDA family's table; each symbol is unique
        // across families (cast_to_*, and the future per-op elementwise /
        // matmul / … symbols), so order is immaterial.
        CUDA_CAST_ENTRY_POINTS
            .iter()
            .find(|(s, _)| *s == symbol)
            .map(|(_, k)| *k)
    }

    fn resolve_fused(&self, _symbol: &str) -> Option<KernelRef> {
        // No fused-op contracts in the CUDA cast corpus — every section is a
        // primitive `op_kind`.
        None
    }
}
