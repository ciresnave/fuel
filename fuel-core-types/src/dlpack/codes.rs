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

// --- FDXScalePlacement (§6.2) ---
/// Scales interleaved in the data block (GGML family).
pub const FDX_SCALE_PLACEMENT_INLINE: u8 = 0;
/// `scale_buffer` is a real buffer-table index (MX / AFFINE_*).
pub const FDX_SCALE_PLACEMENT_SEPARATE_BUFFER: u8 = 1;
/// Scale tensor shape per granularity (broadcast per axis).
pub const FDX_SCALE_PLACEMENT_BROADCAST_PER_AXIS: u8 = 2;

// --- FDXBufferRef.role (§7.2, §6.9.3) ---
pub const FDX_BUFFER_ROLE_DATA: u8 = 0;
pub const FDX_BUFFER_ROLE_SCALE: u8 = 1;
pub const FDX_BUFFER_ROLE_ZERO_POINT: u8 = 2;
pub const FDX_BUFFER_ROLE_BUNDLE_BACKING: u8 = 3;
pub const FDX_BUFFER_ROLE_AUX: u8 = 4;
/// Gather: the physical block pool (§6.9.3).
pub const FDX_BUFFER_ROLE_POOL: u8 = 5;
/// Gather: the `[num_sequences, max_blocks_per_seq]` block-id table (§6.9.3).
pub const FDX_BUFFER_ROLE_BLOCK_TABLE: u8 = 6;
/// Gather: the `[num_sequences]` per-sequence live-length buffer (§6.9.3).
pub const FDX_BUFFER_ROLE_CONTEXT_LENS: u8 = 7;

// --- FDXResidency.tier (§6.6) — three-tier residency ---
pub const FDX_TIER_DEVICE: u8 = 0;
pub const FDX_TIER_HOST: u8 = 1;
pub const FDX_TIER_DISK_MMAP: u8 = 2;

// --- FDXResidency.substrate (§6.0/§6.6) — FDX-owned; pins `SubstrateClass`
// (that source enum is `#[non_exhaustive]`; these codes are NOT its
// discriminants — the §6.0 conversion fn + build-time test is the binding). ---
pub const FDX_SUBSTRATE_HOST_BYTES: u8 = 0;
pub const FDX_SUBSTRATE_CUDA_UNTYPED: u8 = 1;
pub const FDX_SUBSTRATE_VULKAN_BUFFER: u8 = 2;
pub const FDX_SUBSTRATE_METAL_BUFFER: u8 = 3;

// --- FDXResidency.backend_id (§6.0/§6.6) — FDX-owned; pins `BackendId`
// (`Aocl`/`Mkl` already retired from that enum). NOT a `BackendId`
// discriminant; the §6.0 conversion fn + build-time test is the binding. ---
pub const FDX_BACKEND_CPU: u8 = 0;
pub const FDX_BACKEND_CUDA: u8 = 1;
pub const FDX_BACKEND_VULKAN: u8 = 2;
pub const FDX_BACKEND_METAL: u8 = 3;

// --- FDXStorage.class (§6.7) ---
pub const FDX_STORAGE_SHARED: u8 = 0;
pub const FDX_STORAGE_SESSION: u8 = 1;
pub const FDX_STORAGE_TRANSIENT: u8 = 2;

// --- FDX ggml_dtype codes (§6.2) — FDX-owned; mirror `GgmlDType::to_u32`.
// Re-declared here so the conversion fns + mapping test reference FDX symbols,
// not the source enum's `to_u32` (which the §6.0 binding keeps pinned). ---
pub const FDX_GGML_F32: u16 = 0;
pub const FDX_GGML_F16: u16 = 1;
pub const FDX_GGML_Q4_0: u16 = 2;
pub const FDX_GGML_Q4_1: u16 = 3;
pub const FDX_GGML_Q5_0: u16 = 6;
pub const FDX_GGML_Q5_1: u16 = 7;
pub const FDX_GGML_Q8_0: u16 = 8;
pub const FDX_GGML_Q8_1: u16 = 9;
pub const FDX_GGML_Q2K: u16 = 10;
pub const FDX_GGML_Q3K: u16 = 11;
pub const FDX_GGML_Q4K: u16 = 12;
pub const FDX_GGML_Q5K: u16 = 13;
pub const FDX_GGML_Q6K: u16 = 14;
pub const FDX_GGML_Q8K: u16 = 15;
pub const FDX_GGML_BF16: u16 = 30;

// --- FDX logical dtype codes (§6.1 table) — the subset the validators name.
// FDX-owned stable table; values match the §6.1 declaration order. ---
pub const FDX_DTYPE_U8: u16 = 0;
pub const FDX_DTYPE_I8: u16 = 1;
pub const FDX_DTYPE_U32: u16 = 2;
pub const FDX_DTYPE_I16: u16 = 3;
pub const FDX_DTYPE_I32: u16 = 4;
pub const FDX_DTYPE_I64: u16 = 5;
pub const FDX_DTYPE_BF16: u16 = 6;
pub const FDX_DTYPE_F16: u16 = 7;
pub const FDX_DTYPE_F32: u16 = 8;
pub const FDX_DTYPE_F64: u16 = 9;
pub const FDX_DTYPE_F8E4M3: u16 = 10;
pub const FDX_DTYPE_F6E2M3: u16 = 11;
pub const FDX_DTYPE_F6E3M2: u16 = 12;
pub const FDX_DTYPE_F4: u16 = 13;
pub const FDX_DTYPE_F8E8M0: u16 = 14;

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
