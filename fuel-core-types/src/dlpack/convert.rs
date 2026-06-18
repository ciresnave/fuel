//! FDX code-conversion functions — the **binding** that makes FDX the normative
//! owner of the shared dtype/quant/granularity/substrate/backend codes
//! (spec `docs/specs/dlpack-extension.md` §6.0, §6.1, §6.2, §6.6, Appendix B).
//!
//! Every *forward* conversion is an **exhaustive `match`** on the source enum
//! with **no wildcard arm** (and never `as` on the enum's discriminant). That is
//! the §6.0 "reorder breaks the build" guarantee: adding, removing, or reordering
//! a source variant fails to compile *here* rather than silently shifting the
//! stable FDX ABI. The source enums (`DType`, `GgmlDType`, `ScaleGranularity`,
//! `BackendId`, `SubstrateClass`) are explicitly allowed to evolve; the FDX codes
//! in [`super::codes`] are the fixed cross-tool contract, and these `match`-es +
//! the build-time mapping test below are what keep the two pinned together.
//!
//! `BackendId` / `SubstrateClass` are `#[non_exhaustive]`, but this module lives
//! **inside** the defining crate (`fuel-core-types`), so an exhaustive `match`
//! with no `_` arm is permitted — and is exactly what we want: a new variant must
//! break the build here so its FDX code is assigned deliberately (§6.0).
//!
//! Reverse conversions return `Option`: some FDX codes have no Fuel source
//! variant (e.g. the §6.1 microscaling/escape/reserved codes
//! `GENERIC_LOW_BIT_*`, `COMPLEX64`, `BOOL`), and an out-of-range integer is not
//! a valid code. `None` means "no corresponding Fuel value", never a panic (G6).

use super::codes::*;
use super::sidecar::{FDXAffine, FDXAffineTerm, FDXExtent};
use crate::backend::SubstrateClass;
use crate::dtype::DType;
use crate::probe::BackendId;
use crate::quant_scale::ScaleGranularity;
use crate::quantized::GgmlDType;
use crate::shape::Extent;
use crate::symbol::SymId;

// ─────────────────────────────────────────────────────────────────────────────
// DType ⇄ FDX logical-dtype code (§6.1)
// ─────────────────────────────────────────────────────────────────────────────

/// Map a Fuel [`DType`] to its FDX logical-dtype code (`FDX_DTYPE_*`, §6.1).
///
/// **Exhaustive `match`, no wildcard** — adding/removing/reordering a `DType`
/// variant breaks this compile (the §6.0 guarantee). NEVER `d as u16`: the FDX
/// code is the stable contract, decoupled from the source discriminant.
pub fn dtype_to_fdx(d: DType) -> u16 {
    match d {
        DType::U8 => FDX_DTYPE_U8,
        DType::I8 => FDX_DTYPE_I8,
        DType::U32 => FDX_DTYPE_U32,
        DType::I16 => FDX_DTYPE_I16,
        DType::I32 => FDX_DTYPE_I32,
        DType::I64 => FDX_DTYPE_I64,
        DType::BF16 => FDX_DTYPE_BF16,
        DType::F16 => FDX_DTYPE_F16,
        DType::F32 => FDX_DTYPE_F32,
        DType::F64 => FDX_DTYPE_F64,
        DType::F8E4M3 => FDX_DTYPE_F8E4M3,
        DType::F6E2M3 => FDX_DTYPE_F6E2M3,
        DType::F6E3M2 => FDX_DTYPE_F6E3M2,
        DType::F4 => FDX_DTYPE_F4,
        DType::F8E8M0 => FDX_DTYPE_F8E8M0,
    }
}

/// Map an FDX logical-dtype code back to a Fuel [`DType`].
///
/// `None` for codes that have no Fuel `DType` (the §6.1 escape/reserved codes
/// `GENERIC_LOW_BIT_*` / `COMPLEX64` / `BOOL`, the `NONE` sentinel, and any
/// out-of-range integer). All 15 real `DType` codes round-trip.
pub fn fdx_to_dtype(code: u16) -> Option<DType> {
    let d = match code {
        FDX_DTYPE_U8 => DType::U8,
        FDX_DTYPE_I8 => DType::I8,
        FDX_DTYPE_U32 => DType::U32,
        FDX_DTYPE_I16 => DType::I16,
        FDX_DTYPE_I32 => DType::I32,
        FDX_DTYPE_I64 => DType::I64,
        FDX_DTYPE_BF16 => DType::BF16,
        FDX_DTYPE_F16 => DType::F16,
        FDX_DTYPE_F32 => DType::F32,
        FDX_DTYPE_F64 => DType::F64,
        FDX_DTYPE_F8E4M3 => DType::F8E4M3,
        FDX_DTYPE_F6E2M3 => DType::F6E2M3,
        FDX_DTYPE_F6E3M2 => DType::F6E3M2,
        FDX_DTYPE_F4 => DType::F4,
        FDX_DTYPE_F8E8M0 => DType::F8E8M0,
        _ => return None,
    };
    Some(d)
}

// ─────────────────────────────────────────────────────────────────────────────
// GgmlDType ⇄ FDX ggml_dtype code (§6.2)
// ─────────────────────────────────────────────────────────────────────────────

/// Map a [`GgmlDType`] to its FDX `ggml_dtype` code (`FDX_GGML_*`, §6.2).
///
/// **Exhaustive `match`, no wildcard** — a new `GgmlDType` variant breaks this
/// compile. The codes mirror `GgmlDType::to_u32` (the cross-tool ggml numbering:
/// `Q4_0=2`, `Q5_0=6`, `Q4K=12`, `Q8K=15`, `F16=1`, `BF16=30`), but are pinned
/// here by `match` to the FDX symbols, never read off `to_u32` positionally.
pub fn ggml_to_fdx(g: GgmlDType) -> u16 {
    match g {
        GgmlDType::F32 => FDX_GGML_F32,
        GgmlDType::F16 => FDX_GGML_F16,
        GgmlDType::BF16 => FDX_GGML_BF16,
        GgmlDType::Q4_0 => FDX_GGML_Q4_0,
        GgmlDType::Q4_1 => FDX_GGML_Q4_1,
        GgmlDType::Q5_0 => FDX_GGML_Q5_0,
        GgmlDType::Q5_1 => FDX_GGML_Q5_1,
        GgmlDType::Q8_0 => FDX_GGML_Q8_0,
        GgmlDType::Q8_1 => FDX_GGML_Q8_1,
        GgmlDType::Q2K => FDX_GGML_Q2K,
        GgmlDType::Q3K => FDX_GGML_Q3K,
        GgmlDType::Q4K => FDX_GGML_Q4K,
        GgmlDType::Q5K => FDX_GGML_Q5K,
        GgmlDType::Q6K => FDX_GGML_Q6K,
        GgmlDType::Q8K => FDX_GGML_Q8K,
    }
}

/// Map an FDX `ggml_dtype` code back to a [`GgmlDType`].
///
/// `None` for `FDX_QUANT_NONE` (`0xFFFF`, the "not GGML" sentinel) and any code
/// outside the ggml numbering (e.g. the gaps at 4/5 in the ggml type space).
pub fn fdx_to_ggml(code: u16) -> Option<GgmlDType> {
    let g = match code {
        FDX_GGML_F32 => GgmlDType::F32,
        FDX_GGML_F16 => GgmlDType::F16,
        FDX_GGML_BF16 => GgmlDType::BF16,
        FDX_GGML_Q4_0 => GgmlDType::Q4_0,
        FDX_GGML_Q4_1 => GgmlDType::Q4_1,
        FDX_GGML_Q5_0 => GgmlDType::Q5_0,
        FDX_GGML_Q5_1 => GgmlDType::Q5_1,
        FDX_GGML_Q8_0 => GgmlDType::Q8_0,
        FDX_GGML_Q8_1 => GgmlDType::Q8_1,
        FDX_GGML_Q2K => GgmlDType::Q2K,
        FDX_GGML_Q3K => GgmlDType::Q3K,
        FDX_GGML_Q4K => GgmlDType::Q4K,
        FDX_GGML_Q5K => GgmlDType::Q5K,
        FDX_GGML_Q6K => GgmlDType::Q6K,
        FDX_GGML_Q8K => GgmlDType::Q8K,
        _ => return None,
    };
    Some(g)
}

// ─────────────────────────────────────────────────────────────────────────────
// ScaleGranularity ⇄ FDX scale-granularity code (§6.2)
// ─────────────────────────────────────────────────────────────────────────────

/// Map a [`ScaleGranularity`] to its FDX code (`FDX_SCALE_GRAN_*`, §6.2).
///
/// **Exhaustive `match`, no wildcard** — a new `ScaleGranularity` variant breaks
/// this compile. Note FDX additionally owns `PerBlock` (code 3), which has **no**
/// `ScaleGranularity` source variant (it is MX-family-only block grain, §6.2);
/// that code is therefore only producible from FDX-side construction, and
/// [`fdx_to_scale_granularity`] returns `None` for it.
pub fn scale_granularity_to_fdx(g: ScaleGranularity) -> u8 {
    match g {
        ScaleGranularity::PerTensor => FDX_SCALE_GRAN_PER_TENSOR,
        ScaleGranularity::PerToken => FDX_SCALE_GRAN_PER_TOKEN,
        ScaleGranularity::PerChannel => FDX_SCALE_GRAN_PER_CHANNEL,
    }
}

/// Map an FDX scale-granularity code back to a [`ScaleGranularity`].
///
/// `None` for `FDX_SCALE_GRAN_PER_BLOCK` (MX-only, no Fuel variant — §6.2) and
/// any out-of-range value.
pub fn fdx_to_scale_granularity(code: u8) -> Option<ScaleGranularity> {
    let g = match code {
        FDX_SCALE_GRAN_PER_TENSOR => ScaleGranularity::PerTensor,
        FDX_SCALE_GRAN_PER_TOKEN => ScaleGranularity::PerToken,
        FDX_SCALE_GRAN_PER_CHANNEL => ScaleGranularity::PerChannel,
        // FDX_SCALE_GRAN_PER_BLOCK is MX-only with no ScaleGranularity variant.
        _ => return None,
    };
    Some(g)
}

// ─────────────────────────────────────────────────────────────────────────────
// BackendId ⇄ FDX backend code (§6.0 / §6.6)
// ─────────────────────────────────────────────────────────────────────────────

/// Map a [`BackendId`] to its FDX backend code (`FDX_BACKEND_*`, §6.6).
///
/// **Exhaustive `match`, no wildcard** — even though `BackendId` is
/// `#[non_exhaustive]`, we are inside the defining crate, so this is allowed and
/// REQUIRED: a new backend variant must break this compile so its FDX code is
/// assigned in the §6.6 table deliberately (`Aocl`/`Mkl` were already retired,
/// which is exactly the kind of change this `match` is meant to surface).
pub fn backend_to_fdx(b: BackendId) -> u8 {
    match b {
        BackendId::Cpu => FDX_BACKEND_CPU,
        BackendId::Cuda => FDX_BACKEND_CUDA,
        BackendId::Vulkan => FDX_BACKEND_VULKAN,
        BackendId::Metal => FDX_BACKEND_METAL,
    }
}

/// Map an FDX backend code back to a [`BackendId`]. `None` if out of range.
pub fn fdx_to_backend(code: u8) -> Option<BackendId> {
    let b = match code {
        FDX_BACKEND_CPU => BackendId::Cpu,
        FDX_BACKEND_CUDA => BackendId::Cuda,
        FDX_BACKEND_VULKAN => BackendId::Vulkan,
        FDX_BACKEND_METAL => BackendId::Metal,
        _ => return None,
    };
    Some(b)
}

// ─────────────────────────────────────────────────────────────────────────────
// SubstrateClass ⇄ FDX substrate code (§6.0 / §6.6)
// ─────────────────────────────────────────────────────────────────────────────

/// Map a [`SubstrateClass`] to its FDX substrate code (`FDX_SUBSTRATE_*`, §6.6).
///
/// **Exhaustive `match`, no wildcard** — `SubstrateClass` is `#[non_exhaustive]`
/// but we are inside the defining crate, so the exhaustive `match` is allowed and
/// REQUIRED: a new substrate variant breaks this compile so its FDX code is
/// assigned deliberately. `substrate` + `backend_id` + `device_index` reproduce
/// Fuel's `shares_storage` aliasing predicate across the boundary (§6.6).
pub fn substrate_to_fdx(s: SubstrateClass) -> u8 {
    match s {
        SubstrateClass::HostBytes => FDX_SUBSTRATE_HOST_BYTES,
        SubstrateClass::CudaUntyped => FDX_SUBSTRATE_CUDA_UNTYPED,
        SubstrateClass::VulkanBuffer => FDX_SUBSTRATE_VULKAN_BUFFER,
        SubstrateClass::MetalBuffer => FDX_SUBSTRATE_METAL_BUFFER,
    }
}

/// Map an FDX substrate code back to a [`SubstrateClass`]. `None` if out of range.
pub fn fdx_to_substrate(code: u8) -> Option<SubstrateClass> {
    let s = match code {
        FDX_SUBSTRATE_HOST_BYTES => SubstrateClass::HostBytes,
        FDX_SUBSTRATE_CUDA_UNTYPED => SubstrateClass::CudaUntyped,
        FDX_SUBSTRATE_VULKAN_BUFFER => SubstrateClass::VulkanBuffer,
        FDX_SUBSTRATE_METAL_BUFFER => SubstrateClass::MetalBuffer,
        _ => return None,
    };
    Some(s)
}

// ─────────────────────────────────────────────────────────────────────────────
// Extent ⇄ FDXExtent (Appendix B)
// ─────────────────────────────────────────────────────────────────────────────

/// Map a Fuel [`Extent`] to an [`FDXExtent`] (Appendix B).
///
/// **Exhaustive `match`, no wildcard** — a new `Extent` variant breaks this
/// compile. Only the two Fuel `Extent` variants are producible here:
///
/// - [`Extent::Scalar`]`(v)` → `kind = 0` (Scalar), `min == capacity == v`,
///   `sym_id = FDX_SYM_NONE`.
/// - [`Extent::Range`]`{min, max, sym}` → `kind = 1` (Range), `capacity = max`
///   (what `dims()` reports / strides key to — P4), `sym_id = sym`.
///
/// The third FDX extent kind, `AFFINE` (`kind = 2`, §6.4.2), has **no** Fuel
/// `Extent` source variant — it is produced directly from a symbolic-affine
/// combination, not from this enum — so it is never emitted here and
/// [`fdx_to_extent`] returns `None` for it.
///
/// The `affine`/`cap_kind` sub-block is zeroed (the Scalar/Range-degenerate form
/// required by validator V7), and `sym_scope` is left `0` (`InputDetermined`,
/// the advisory default).
pub fn extent_to_fdx(e: Extent) -> FDXExtent {
    let zero_affine = FDXAffine {
        c0: 0,
        term_count: 0,
        _pad: [0; 7],
        terms: [FDXAffineTerm {
            coeff: 0,
            sym_id: FDX_SYM_NONE,
            _pad: 0,
        }; FDX_AFFINE_MAX_TERMS],
    };
    let (kind, min, capacity, sym_id) = match e {
        Extent::Scalar(v) => (
            FDX_EXTENT_SCALAR as u8,
            v as u64,
            v as u64,
            FDX_SYM_NONE,
        ),
        Extent::Range { min, max, sym } => {
            (FDX_EXTENT_RANGE as u8, min as u64, max as u64, sym.0)
        }
    };
    FDXExtent {
        kind,
        _pad: [0; 3],
        min,
        capacity,
        sym_id,
        sym_scope: 0,
        _pad2: [0; 3],
        cap_kind: FDX_CAP_KIND_EXPLICIT as u8,
        _pad3: [0; 3],
        _pad4: 0,
        affine: zero_affine,
        reserved: [0; 2],
    }
}

/// Map an [`FDXExtent`] back to a Fuel [`Extent`].
///
/// `None` for `kind = AFFINE` (no Fuel `Extent` variant — §6.4.2) and any
/// unknown `kind` byte. Scalar/Range round-trip (a `usize` narrowing of `u64`
/// is performed; the realize-time u64↔usize OOB guard is the validators' job,
/// §6.4 / V14 — this fn is the plain code/field mapping).
pub fn fdx_to_extent(x: &FDXExtent) -> Option<Extent> {
    match x.kind as u16 {
        FDX_EXTENT_SCALAR => Some(Extent::Scalar(x.capacity as usize)),
        FDX_EXTENT_RANGE => Some(Extent::Range {
            min: x.min as usize,
            max: x.capacity as usize,
            sym: SymId(x.sym_id),
        }),
        // FDX_EXTENT_AFFINE has no Fuel `Extent` variant; unknown kinds → None.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── DType ─────────────────────────────────────────────────────────────

    /// Representative of EVERY `DType` variant. Compile-time exhaustiveness in
    /// `dtype_to_fdx` already guards additions; this list guards value
    /// correctness (the "mapping is total" test below iterates it).
    const ALL_DTYPES: [DType; 15] = [
        DType::U8,
        DType::I8,
        DType::U32,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::BF16,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::F8E4M3,
        DType::F6E2M3,
        DType::F6E3M2,
        DType::F4,
        DType::F8E8M0,
    ];

    #[test]
    fn dtype_roundtrips_for_every_variant() {
        for d in ALL_DTYPES {
            let code = dtype_to_fdx(d);
            assert_eq!(
                fdx_to_dtype(code),
                Some(d),
                "DType {d:?} (code {code}) must round-trip"
            );
        }
    }

    #[test]
    fn dtype_anchor_codes_match_spec_6_1() {
        // Anchors pinned against the §6.1 logical-dtype table.
        assert_eq!(dtype_to_fdx(DType::U8), FDX_DTYPE_U8);
        assert_eq!(dtype_to_fdx(DType::U8), 0);
        assert_eq!(dtype_to_fdx(DType::I8), 1);
        assert_eq!(dtype_to_fdx(DType::F32), FDX_DTYPE_F32);
        assert_eq!(dtype_to_fdx(DType::F32), 8);
        assert_eq!(dtype_to_fdx(DType::F64), 9);
        // The sub-byte / microscaling ones the spec calls out explicitly.
        assert_eq!(dtype_to_fdx(DType::F8E4M3), 10);
        assert_eq!(dtype_to_fdx(DType::F6E2M3), 11);
        assert_eq!(dtype_to_fdx(DType::F6E3M2), 12);
        assert_eq!(dtype_to_fdx(DType::F4), 13);
        assert_eq!(dtype_to_fdx(DType::F8E8M0), 14);
    }

    #[test]
    fn dtype_codes_have_no_duplicates() {
        let mut codes: Vec<u16> = ALL_DTYPES.iter().copied().map(dtype_to_fdx).collect();
        let n = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), n, "FDX_DTYPE_* set must have no duplicate codes");
    }

    #[test]
    fn fdx_to_dtype_rejects_escape_and_reserved_codes() {
        // §6.1 escape/reserved codes have no Fuel DType.
        assert_eq!(fdx_to_dtype(0x0100), None); // GENERIC_LOW_BIT_INT
        assert_eq!(fdx_to_dtype(0x0101), None); // GENERIC_LOW_BIT_FLOAT
        assert_eq!(fdx_to_dtype(0x0200), None); // COMPLEX64 (reserved)
        assert_eq!(fdx_to_dtype(0x0201), None); // BOOL (reserved)
        assert_eq!(fdx_to_dtype(FDX_DTYPE_NONE), None); // 0xFFFF NONE
        assert_eq!(fdx_to_dtype(15), None); // first unused contiguous code
    }

    // ── GgmlDType ─────────────────────────────────────────────────────────

    const ALL_GGML: [GgmlDType; 15] = [
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::BF16,
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q8_1,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8K,
    ];

    #[test]
    fn ggml_roundtrips_for_every_variant() {
        for g in ALL_GGML {
            let code = ggml_to_fdx(g);
            assert_eq!(
                fdx_to_ggml(code),
                Some(g),
                "GgmlDType {g:?} (code {code}) must round-trip"
            );
        }
    }

    #[test]
    fn ggml_codes_match_source_numbering_and_anchors() {
        // FDX ggml codes mirror GgmlDType::to_u32 (the cross-tool numbering).
        for g in ALL_GGML {
            assert_eq!(
                ggml_to_fdx(g) as u32,
                g.to_u32(),
                "FDX ggml code must equal GgmlDType::to_u32 for {g:?}"
            );
        }
        // The §6.2-named anchors.
        assert_eq!(ggml_to_fdx(GgmlDType::Q4K), FDX_GGML_Q4K);
        assert_eq!(ggml_to_fdx(GgmlDType::Q4K), 12);
        assert_eq!(ggml_to_fdx(GgmlDType::Q4_0), 2);
        assert_eq!(ggml_to_fdx(GgmlDType::Q5_0), 6);
        assert_eq!(ggml_to_fdx(GgmlDType::Q8K), 15);
        assert_eq!(ggml_to_fdx(GgmlDType::F16), 1);
        assert_eq!(ggml_to_fdx(GgmlDType::BF16), 30);
    }

    #[test]
    fn fdx_to_ggml_rejects_gaps_and_none() {
        assert_eq!(fdx_to_ggml(FDX_QUANT_NONE), None); // 0xFFFF
        assert_eq!(fdx_to_ggml(4), None); // gap in ggml numbering
        assert_eq!(fdx_to_ggml(5), None); // gap in ggml numbering
        assert_eq!(fdx_to_ggml(16), None);
    }

    // ── ScaleGranularity ──────────────────────────────────────────────────

    const ALL_GRAN: [ScaleGranularity; 3] = [
        ScaleGranularity::PerTensor,
        ScaleGranularity::PerToken,
        ScaleGranularity::PerChannel,
    ];

    #[test]
    fn scale_granularity_roundtrips_for_every_variant() {
        for g in ALL_GRAN {
            let code = scale_granularity_to_fdx(g);
            assert_eq!(fdx_to_scale_granularity(code), Some(g), "gran {g:?} round-trip");
        }
        assert_eq!(scale_granularity_to_fdx(ScaleGranularity::PerTensor), 0);
        assert_eq!(scale_granularity_to_fdx(ScaleGranularity::PerToken), 1);
        assert_eq!(scale_granularity_to_fdx(ScaleGranularity::PerChannel), 2);
    }

    #[test]
    fn per_block_granularity_has_no_fuel_variant() {
        // §6.2: PerBlock (code 3) is MX-only — no ScaleGranularity source.
        assert_eq!(fdx_to_scale_granularity(FDX_SCALE_GRAN_PER_BLOCK), None);
        assert_eq!(fdx_to_scale_granularity(4), None);
    }

    // ── BackendId ─────────────────────────────────────────────────────────

    const ALL_BACKENDS: [BackendId; 4] =
        [BackendId::Cpu, BackendId::Cuda, BackendId::Vulkan, BackendId::Metal];

    #[test]
    fn backend_roundtrips_for_every_variant() {
        for b in ALL_BACKENDS {
            let code = backend_to_fdx(b);
            assert_eq!(fdx_to_backend(code), Some(b), "backend {b:?} round-trip");
        }
        assert_eq!(backend_to_fdx(BackendId::Cpu), 0);
        assert_eq!(backend_to_fdx(BackendId::Cuda), 1);
        assert_eq!(backend_to_fdx(BackendId::Vulkan), 2);
        assert_eq!(backend_to_fdx(BackendId::Metal), 3);
        assert_eq!(fdx_to_backend(4), None);
    }

    // ── SubstrateClass ────────────────────────────────────────────────────

    const ALL_SUBSTRATES: [SubstrateClass; 4] = [
        SubstrateClass::HostBytes,
        SubstrateClass::CudaUntyped,
        SubstrateClass::VulkanBuffer,
        SubstrateClass::MetalBuffer,
    ];

    #[test]
    fn substrate_roundtrips_for_every_variant() {
        for s in ALL_SUBSTRATES {
            let code = substrate_to_fdx(s);
            assert_eq!(fdx_to_substrate(code), Some(s), "substrate {s:?} round-trip");
        }
        assert_eq!(substrate_to_fdx(SubstrateClass::HostBytes), 0);
        assert_eq!(substrate_to_fdx(SubstrateClass::CudaUntyped), 1);
        assert_eq!(substrate_to_fdx(SubstrateClass::VulkanBuffer), 2);
        assert_eq!(substrate_to_fdx(SubstrateClass::MetalBuffer), 3);
        assert_eq!(fdx_to_substrate(4), None);
    }

    // ── Extent ────────────────────────────────────────────────────────────

    #[test]
    fn extent_scalar_roundtrips() {
        let e = Extent::Scalar(4096);
        let x = extent_to_fdx(e);
        assert_eq!(x.kind as u16, FDX_EXTENT_SCALAR);
        assert_eq!(x.capacity, 4096);
        assert_eq!(x.min, 4096);
        assert_eq!(x.sym_id, FDX_SYM_NONE);
        assert_eq!(x.affine.term_count, 0);
        assert_eq!(fdx_to_extent(&x), Some(e));
    }

    #[test]
    fn extent_range_roundtrips() {
        let e = Extent::Range { min: 1, max: 8192, sym: SymId(7) };
        let x = extent_to_fdx(e);
        assert_eq!(x.kind as u16, FDX_EXTENT_RANGE);
        assert_eq!(x.min, 1);
        assert_eq!(x.capacity, 8192); // capacity == max (P4); strides key to it
        assert_eq!(x.sym_id, 7);
        assert_eq!(fdx_to_extent(&x), Some(e));
    }

    #[test]
    fn affine_extent_has_no_fuel_variant() {
        // §6.4.2: AFFINE has no Fuel `Extent` source; reverse → None.
        let mut x = extent_to_fdx(Extent::Scalar(1));
        x.kind = FDX_EXTENT_AFFINE as u8;
        assert_eq!(fdx_to_extent(&x), None);
        // Unknown kind byte → None (no panic).
        x.kind = 99;
        assert_eq!(fdx_to_extent(&x), None);
    }

    // ── "mapping is total" (value correctness over every variant) ─────────

    #[test]
    fn every_source_variant_maps_to_a_distinct_real_code() {
        // DType: 15 variants → 15 distinct non-NONE codes.
        let dt: Vec<u16> = ALL_DTYPES.iter().copied().map(dtype_to_fdx).collect();
        for &c in &dt {
            assert_ne!(c, FDX_DTYPE_NONE);
            assert!(fdx_to_dtype(c).is_some());
        }
        assert_eq!({ let mut v = dt.clone(); v.sort_unstable(); v.dedup(); v.len() }, dt.len());

        // GgmlDType: 15 variants → 15 distinct codes, none == NONE sentinel.
        let gg: Vec<u16> = ALL_GGML.iter().copied().map(ggml_to_fdx).collect();
        for &c in &gg {
            assert_ne!(c, FDX_QUANT_NONE);
            assert!(fdx_to_ggml(c).is_some());
        }
        assert_eq!({ let mut v = gg.clone(); v.sort_unstable(); v.dedup(); v.len() }, gg.len());

        // ScaleGranularity / BackendId / SubstrateClass: total + distinct.
        let gr: Vec<u8> = ALL_GRAN.iter().copied().map(scale_granularity_to_fdx).collect();
        assert!(gr.iter().all(|&c| fdx_to_scale_granularity(c).is_some()));
        let bk: Vec<u8> = ALL_BACKENDS.iter().copied().map(backend_to_fdx).collect();
        assert!(bk.iter().all(|&c| fdx_to_backend(c).is_some()));
        let su: Vec<u8> = ALL_SUBSTRATES.iter().copied().map(substrate_to_fdx).collect();
        assert!(su.iter().all(|&c| fdx_to_substrate(c).is_some()));
    }
}
