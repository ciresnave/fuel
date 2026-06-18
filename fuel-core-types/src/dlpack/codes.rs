//! FDX-owned stable code tables, magic, version, flags, and sentinels — the
//! **normative source** for every shared dtype/quant/granularity/substrate/
//! gather/extent code (spec §5.2, §6.0, Appendix A). The sibling FKC kernel-
//! contract format references these by symbol and re-numbers nothing; a
//! build-time mapping test (added with the conversion fns in the next slice)
//! is the binding, so a source-enum reorder breaks the build rather than
//! silently shifting the ABI.
//!
//! Field widths are chosen to match the `FDXSidecar` struct fields they
//! populate (added in the sidecar slice): the `0xFFFF` sentinels are `u16`,
//! the `0xFFFF_FFFF` sentinels and `FDXSidecar.flags` are `u32`, scale
//! granularity is `u8`.

// --- magic / schema version (§5.2) ---
/// `"FDX\0"`.
pub const FDX_MAGIC: u32 = 0x4644_5800;
/// FDX **schema** major — independent of the DLPack ABI version.
pub const FDX_VERSION_1: u32 = 1;
pub const FDX_VERSION_MAX: u32 = 1;

// --- sentinels (Appendix A) ---
pub const FDX_SYM_NONE: u32 = 0xFFFF_FFFF;
pub const FDX_SESSION_NONE: u32 = 0;
pub const FDX_DTYPE_NONE: u16 = 0xFFFF;
pub const FDX_BUFFER_INLINE: u32 = 0xFFFF_FFFF;
pub const FDX_BUFFER_NONE: u32 = 0xFFFF_FFFE;
pub const FDX_BLOCK_UNMAPPED: u32 = 0xFFFF_FFFF;

// --- FDXSidecar.flags (§5.2 authoritative bit-allocation table; `flags` is u32) ---
pub const FDX_FLAG_HAS_DTYPE_EXT: u32 = 1 << 0;
pub const FDX_FLAG_HAS_QUANT: u32 = 1 << 1;
pub const FDX_FLAG_HAS_SYMBOLIC: u32 = 1 << 2;
pub const FDX_FLAG_HAS_TILING: u32 = 1 << 3;
pub const FDX_FLAG_IS_BUNDLE: u32 = 1 << 4;
/// Base bytes alone are NOT a usable tensor (quant/sub-byte/unbacked tail) —
/// drives the refuse-or-dequantize producer policy (§9).
pub const FDX_FLAG_MEANING_REQUIRES_EXT: u32 = 1 << 5;
pub const FDX_FLAG_READ_ONLY: u32 = 1 << 6;
/// Base bytes are a physical block pool re-interpreted via a block table
/// (§6.9). Implies [`FDX_FLAG_MEANING_REQUIRES_EXT`] (validator V18).
pub const FDX_FLAG_HAS_GATHER: u32 = 1 << 7;
/// ≥1 extent is `kind = AFFINE` (§6.4). Advisory fast-reject; the per-extent
/// `kind` byte is authoritative.
pub const FDX_FLAG_HAS_AFFINE_EXTENT: u32 = 1 << 8;
// bits 9..=31 reserved (0); the next addition takes bit 9 from this table.

/// Every allocated `FDX_FLAG_*` bit, in allocation order. The single source
/// for the no-collision build-time test; a future flag appends here.
pub const FDX_FLAG_ALL: [u32; 9] = [
    FDX_FLAG_HAS_DTYPE_EXT,
    FDX_FLAG_HAS_QUANT,
    FDX_FLAG_HAS_SYMBOLIC,
    FDX_FLAG_HAS_TILING,
    FDX_FLAG_IS_BUNDLE,
    FDX_FLAG_MEANING_REQUIRES_EXT,
    FDX_FLAG_READ_ONLY,
    FDX_FLAG_HAS_GATHER,
    FDX_FLAG_HAS_AFFINE_EXTENT,
];

// --- FDXExtent.kind (§6.4) ---
pub const FDX_EXTENT_SCALAR: u16 = 0;
pub const FDX_EXTENT_RANGE: u16 = 1;
pub const FDX_EXTENT_AFFINE: u16 = 2;
/// Affine term-count cap; overflow is a typed error (§6.4).
pub const FDX_AFFINE_MAX_TERMS: usize = 4;
/// `FDXExtent.cap_kind` (affine): explicit capacity (v1).
pub const FDX_CAP_KIND_EXPLICIT: u16 = 0;
/// Reserved for a future capacity-from-affine-max form.
pub const FDX_CAP_KIND_AFFINE_MAX: u16 = 1;

// --- FDXIndexedResidency.kind (gather, §6.9) ---
pub const FDX_GATHER_NONE: u16 = 0;
pub const FDX_GATHER_PAGED_BLOCKS: u16 = 1;

// --- FDXQuant.family (§6.2) — FDX is the normative owner of these codes ---
pub const FDX_QUANT_NONE: u16 = 0xFFFF;
/// Baked INLINE scales; `ggml_dtype` ONLY, never `PerBlock`/separate scale.
pub const FDX_QUANT_GGML_BLOCK: u16 = 0;
/// F8E8M0 per-block scale; the **sole** `PerBlock` user.
pub const FDX_QUANT_MX: u16 = 1;
pub const FDX_QUANT_AFFINE_INT: u16 = 2;
/// Per-tensor/token/channel affine float quant.
pub const FDX_QUANT_AFFINE_FLOAT: u16 = 3;
/// nf4/QLoRA: low-bit data + a SEPARATE block-shaped scale operand (§6.2).
pub const FDX_QUANT_AFFINE_BLOCK: u16 = 4;

// --- FDXScaleGranularity (§6.2) ---
pub const FDX_SCALE_GRAN_PER_TENSOR: u8 = 0;
pub const FDX_SCALE_GRAN_PER_TOKEN: u8 = 1;
pub const FDX_SCALE_GRAN_PER_CHANNEL: u8 = 2;
/// `PerBlock` is MX-family only (§6.2).
pub const FDX_SCALE_GRAN_PER_BLOCK: u8 = 3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fdx_flag_bits_are_unique_single_bits() {
        let mut seen = 0u32;
        for &f in &FDX_FLAG_ALL {
            assert_eq!(f.count_ones(), 1, "flag {f:#x} must be exactly one bit");
            assert_eq!(seen & f, 0, "flag {f:#x} collides with an earlier flag");
            seen |= f;
        }
        assert_eq!(seen.count_ones() as usize, FDX_FLAG_ALL.len());
        // gather implies meaning-requires-ext per §6.9/V18; both bits exist.
        assert_ne!(FDX_FLAG_HAS_GATHER & FDX_FLAG_MEANING_REQUIRES_EXT, FDX_FLAG_HAS_GATHER);
    }

    #[test]
    fn fdx_quant_family_codes_are_distinct() {
        let fams = [
            FDX_QUANT_GGML_BLOCK,
            FDX_QUANT_MX,
            FDX_QUANT_AFFINE_INT,
            FDX_QUANT_AFFINE_FLOAT,
            FDX_QUANT_AFFINE_BLOCK,
        ];
        for (i, a) in fams.iter().enumerate() {
            for b in &fams[i + 1..] {
                assert_ne!(a, b, "quant family codes must be distinct");
            }
            assert_ne!(*a, FDX_QUANT_NONE, "a real family must not equal NONE");
        }
    }

    #[test]
    fn fdx_constant_sentinels() {
        assert_eq!(FDX_MAGIC, 0x4644_5800);
        assert_eq!(FDX_AFFINE_MAX_TERMS, 4);
        assert_eq!(FDX_DTYPE_NONE, 0xFFFF);
        assert_eq!(FDX_SYM_NONE, 0xFFFF_FFFF);
        assert_eq!(FDX_BUFFER_INLINE, 0xFFFF_FFFF);
        // PerBlock is MX-only — record the invariant the validators enforce.
        assert_eq!(FDX_SCALE_GRAN_PER_BLOCK, 3);
    }
}
