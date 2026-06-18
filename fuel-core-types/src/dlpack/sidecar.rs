//! FDX sidecar `#[repr(C)]` POD structs — the optional, versioned sidecar
//! communicated alongside a standard `DLTensor` (spec §5.3, §6.1–§6.9, §7).
//!
//! Transcribed faithfully from `docs/specs/dlpack-extension.md`. This is **ABI
//! code**: field names, types, order, and `reserved[…]`/`_pad*` arrays match the
//! spec's Rust struct blocks exactly, so the layout matches the co-maintained
//! `fuel_dlpack_ext.h` C header byte-for-byte on a 64-bit little-endian host
//! (the v1 target, §5).
//!
//! Every variable-length array is referenced by a `(count, *const T)` pair for
//! the live in-memory form; the **serialized** form replaces the pointer with a
//! byte offset relative to the sidecar blob start (P7). The pointer fields are
//! never serialized.
//!
//! All FDX codes/sentinels/flags come from [`super::codes`]; the standard DLPack
//! structs come from [`super::abi`]. Nothing here re-defines either.

use super::abi::DLDevice;
use super::codes::FDX_AFFINE_MAX_TERMS;
use core::ffi::c_void;

/// Sub-byte / microscaling dtype descriptor (spec §6.1). Carries the *true*
/// element type when the base `DLTensor.dtype` is the honesty-preserving `uint8`
/// stand-in. Valid iff `FDX_FLAG_HAS_DTYPE_EXT`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FDXDTypeExt {
    /// FDX-owned stable code for the logical element type (§6.1 table); also
    /// covers a "general low-bit int/float" escape. NOT a `DType` discriminant.
    pub logical_dtype: u16,
    /// True bit-width per logical element: 4, 6, 8, 16, 32, 64. NEVER 0. For
    /// sub-byte Fuel dtypes whose `DType::size_in_bytes() == 0`, this is the
    /// authoritative size source.
    pub bit_width: u16,
    /// How sub-byte elements are packed into bytes (§6.1 `FDXPacking` table).
    pub packing: u8,
    /// Lanes (SIMD width within one logical element); 1 for scalar dtypes.
    pub lanes: u8,
    /// Endianness of multi-bit-but-sub-byte packing within a byte: 0 = LSB-first
    /// (element 0 in low nibble), 1 = MSB-first. Matches GGUF/MX conventions.
    pub sub_byte_bit_order: u8,
    pub _pad: u8,
    pub reserved: [u32; 2],
}

/// Parametric quantization block layout (spec §6.2). Describes GGUF/GGML-style,
/// OCP-microscaling (MX)-style, and affine-int quant by parameters. Valid iff
/// `FDX_FLAG_HAS_QUANT`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FDXQuant {
    /// Quant family/regime (§6.2 `FDX_QUANT_*` table). `FDX_QUANT_NONE` when no
    /// quant.
    pub family: u16,
    /// For the GGML family: the `GgmlDType` code (FDX-owned). `0xFFFF` if not
    /// GGML.
    pub ggml_dtype: u16,

    /// Block geometry, PARAMETRIC. `block_shape[0..block_ndim]` gives the block
    /// extent along each quantized axis. 0 ⇒ "not blocked / per-tensor".
    pub block_ndim: u8,
    pub _pad0: [u8; 3],
    pub block_shape: [u32; 4],
    /// Which base axes the block tiles, parallel to `block_shape`. -1 unused.
    pub block_axes: [i32; 4],

    /// Packing order of the quantized payload relative to the block
    /// (`FDXPackOrder`).
    pub pack_order: u8,
    pub _pad1: [u8; 3],

    /// SCALE descriptor.
    pub scale_present: u8,
    pub scale_dtype: u16,
    pub scale_placement: u8,
    pub scale_granularity: u8,
    pub _pad2: [u8; 3],
    /// Buffer-table index of the scale tensor (§7.4). `FDX_BUFFER_INLINE` if
    /// INLINE (scales interleaved in the data block, GGML family).
    pub scale_buffer: u32,

    /// ZERO-POINT descriptor (affine-int). `zp_present == 0` for symmetric quant.
    pub zp_present: u8,
    pub zp_dtype: u16,
    pub _pad3: u8,
    pub zp_buffer: u32,

    /// For dynamic affine quant: the activation×weight pairing this tensor
    /// participates in. `role` tells whether THIS tensor is activation or weight.
    pub scale_pair_act: u8,
    pub scale_pair_weight: u8,
    pub role: u8,
    pub _pad4: u8,

    pub reserved: [u32; 6],
}

/// One affine term `coeff * sym_id` (spec §6.4). `#[repr(C)]` POD, EXACTLY 16
/// bytes (spec-stated).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FDXAffineTerm {
    /// Signed integer coefficient (i64). Negative coeffs are allowed (e.g.
    /// `capacity - cached_len`); they do NOT relax the per-symbol bounds of the
    /// referenced syms (§6.4 "negative-coeff note").
    pub coeff: i64,
    /// Base `SymId(u32)` bound in the `FDXSymEnv`. `FDX_SYM_NONE` ⇒ unused slot
    /// (then `coeff == 0`). MUST be a BASE symbol — never another affine result
    /// (no nesting).
    pub sym_id: u32,
    pub _pad: u32,
}

/// `value = c0 + Σ_{i<term_count} terms[i].coeff * resolve(terms[i].sym_id)`
/// (spec §6.4). Fixed-capacity (`FDX_AFFINE_MAX_TERMS = 4`), inline, POD.
/// EXACTLY 80 bytes (spec-stated).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FDXAffine {
    /// Constant term (i64).
    pub c0: i64,
    /// Active term count, `0..=FDX_AFFINE_MAX_TERMS`.
    pub term_count: u8,
    pub _pad: [u8; 7],
    /// Slots `>= term_count` are zeroed (`sym_id = FDX_SYM_NONE`, `coeff = 0`).
    pub terms: [FDXAffineTerm; FDX_AFFINE_MAX_TERMS],
}

/// Symbolic / dynamic extent — live-vs-capacity (spec §6.4). One entry per base
/// axis when `extents_count == base.ndim`.
///
/// **Layout rule (load-bearing).** The leading bytes are frozen at the original
/// §6.4 field order: `kind` at offset 0, `min` at 8, `capacity` at 16, `sym_id`
/// at 24, `sym_scope` at offset 28 (with `_pad2[3]` at 29..31). The affine
/// machinery (`cap_kind`, the `affine` sub-block) is appended strictly after
/// offset 32. `FDXExtent` is both a variable-length array element (where its
/// `sizeof` is the `extents[]` stride) AND an inline array member of
/// `FDXIndexedResidency` (`logical_extents: [FDXExtent; 6]`), so its size and
/// every field offset are pinned in the tests below.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FDXExtent {
    /// Axis kind: 0 = Scalar, 1 = Range, 2 = Affine. Offset 0.
    pub kind: u8,
    pub _pad: [u8; 3],
    /// Range only: the lower bound of the live value. Scalar: equals capacity.
    /// Affine: the GUARANTEED minimum (V14 lower bound). Offset 8.
    pub min: u64,
    /// CAPACITY (== base `DLTensor.shape[i]`). Strides in the base are keyed to
    /// THIS (P4). Offset 16.
    pub capacity: u64,
    /// Range only: the `SymId` of the live value (resolved per call via SymEnv).
    /// Scalar & Affine: `FDX_SYM_NONE`. Offset 24.
    pub sym_id: u32,
    /// Scope hint: 0 = InputDetermined, 1 = DataDetermined, 2 = SessionScoped.
    /// Advisory. FROZEN at offset 28 (do NOT move — cross-version byte compat).
    pub sym_scope: u8,
    pub _pad2: [u8; 3], // offsets 29..31
    /// AFFINE only (kind == 2): how `capacity` is determined. 0 = EXPLICIT (v1),
    /// 1 = AFFINE_MAX (RESERVED). MUST be 0 for kind ∈ {Scalar, Range} (V7).
    /// Offset 32.
    pub cap_kind: u8,
    pub _pad3: [u8; 3], // offsets 33..35
    pub _pad4: u32,     // offsets 36..39 (8-byte align the affine sub-block)
    /// AFFINE only (kind == 2): the combination. All-zero for Scalar/Range.
    /// Carried inline (POD, no pointer). Offset 40.
    pub affine: FDXAffine,
    pub reserved: [u32; 2],
}

/// Tiling / alignment hints (spec §6.5). Valid iff `FDX_FLAG_HAS_TILING`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FDXTiling {
    /// Required base-address alignment in bytes (mirrors
    /// `BackendCapabilities::required_alignment`). 0 = unspecified.
    pub alignment_bytes: u32,
    /// Smallest addressable unit in bits. 0 = unspecified (assume 8).
    pub access_granularity_bits: u32,
    /// Optional inner tile shape the buffer is blocked into. `tile_ndim == 0` ⇒
    /// no tiling.
    pub tile_ndim: u8,
    pub _pad: [u8; 7],
    pub tile_shape: [u32; 4],
    pub reserved: [u32; 4],
}

/// Residency / substrate, finer than DLPack's `(device_type, device_id)`
/// (spec §6.6).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FDXResidency {
    /// Three-tier residency: 0 = Device, 1 = Host, 2 = DiskMmap.
    pub tier: u8,
    /// FDX substrate class (§6.0): 0 = HostBytes, 1 = CudaUntyped,
    /// 2 = VulkanBuffer, 3 = MetalBuffer. Decides aliasing.
    pub substrate: u8,
    /// FDX backend code (§6.0): 0 = Cpu, 1 = Cuda, 2 = Vulkan, 3 = Metal.
    pub backend_id: u8,
    pub _pad: u8,
    /// Device ordinal within the backend.
    pub device_index: u32,
    /// Whether this buffer is a zero-copy view into a larger mmap'd region (1)
    /// vs an owned allocation (0).
    pub is_mmap_view: u8,
    pub _pad2: [u8; 7],
    pub reserved: [u32; 4],
}

/// Storage class + session identity (spec §6.7).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FDXStorage {
    /// Storage class: 0 = Shared, 1 = Session, 2 = Transient.
    pub class: u8,
    pub _pad: [u8; 3],
    pub _pad_align: u32,
    /// Session identity for class = Session. `FDX_SESSION_NONE` (0) when not
    /// session-scoped.
    pub session_id: u64,
    pub reserved: [u32; 4],
}

/// Multi-output bundle sub-view (spec §6.8). When `FDX_FLAG_IS_BUNDLE` is set,
/// the base `DLTensor` describes the whole bundle buffer as `uint8` and
/// `views[0..views_count]` partition it.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FDXOutputView {
    /// Byte offset into the bundle buffer (buffer-table index 0).
    pub byte_offset: u64,
    pub len_elements: u64,
    /// FDX logical dtype code for this slot.
    pub dtype: u16,
    pub _pad: [u8; 2],
    pub ndim: u32,
    pub shape: [u64; 6], // matches Fuel DimVec inline capacity (6)
    /// Slot strides, length `ndim` — ALWAYS explicit (§3.2); int64, negatives
    /// first-class (§3.2.1 / V13).
    pub strides: [i64; 6],
    /// Stable slot name hash (FNV-1a of the slot name). 0 = anonymous.
    pub name_hash: u64,
    pub reserved: [u32; 4],
}

/// Logical → physical block mapping for a paged pool (spec §6.9.2). Batched over
/// sequences. Frozen-size; grows only via `reserved`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FDXBlockTable {
    /// Buffer-table index (§7.4) of the block-id table buffer (role =
    /// `FDX_BUFFER_BLOCK_TABLE`), a `[num_sequences, max_blocks_per_seq]` tensor
    /// of physical block ids. P7 — index, not ptr.
    pub table_buffer: u32,
    /// FDX logical dtype code of a block id. PINNED to U32 in v1 (code 2).
    pub id_dtype: u16,
    pub _pad0: u16,
    /// Slots per sequence (columns) = `max_blocks_per_seq`. Widened to u32.
    pub max_blocks_per_seq: u32,
    /// Sentinel block id meaning "unmapped / not yet allocated".
    /// `FDX_BLOCK_UNMAPPED` (0xFFFFFFFF) by default.
    pub unmapped_sentinel: u32,
    /// Layout flags: bit0 = ids sorted within a row (advisory); bit1 = table is
    /// shared/read-only across the call. 0 = no claims.
    pub layout_flags: u32,
    pub reserved: [u32; 4],
}

/// GATHER / INDEXED-RESIDENCY descriptor (spec §6.9.2): re-interprets a
/// contiguous physical BLOCK POOL (the honest `uint8` base, §3) as a
/// logically-gathered tensor via a per-sequence block table. Models a vLLM-style
/// paged KV cache as a SINGLE FDX tensor. Embedded by value in `FDXSidecar`,
/// valid iff `FDX_FLAG_HAS_GATHER`. Frozen-size; grows only via `reserved`.
///
/// `FDXExtent` is doubly load-bearing here: `logical_extents: [FDXExtent; 6]` is
/// an inline member, so any change to `sizeof(FDXExtent)` shifts every field
/// after `logical_extents`. The tests pin `offset_of!` for the trailing fields.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FDXIndexedResidency {
    /// Gather kind: 0 = `FDX_GATHER_NONE` (absent), 1 = `FDX_GATHER_PAGED_BLOCKS`.
    pub kind: u8,
    pub _pad0: [u8; 3],

    // ── PHYSICAL POOL geometry ──────────────────────────────────────────────
    /// Number of fixed-size blocks in the pool (derived = `k_cache.shape[0]`).
    pub num_blocks: u64,
    /// Tokens (logical positions) per physical block. NEVER 0 (V18).
    pub block_size: u64,

    /// Buffer-table index (§7.4) of the physical block-pool buffer. Role =
    /// `FDX_BUFFER_POOL`. Conventionally 0 (the base data buffer). P7 — index.
    pub pool_buffer: u32,
    pub _pad1: u32,

    /// PHYSICAL (pool) typed shape mirroring `[num_blocks, block_size, Hkv, D]`.
    pub physical_ndim: u8,
    pub _pad2: [u8; 7],
    pub physical_shape: [u64; 6],
    /// Physical pool strides in TYPED ELEMENTS (NOT bytes), length
    /// `physical_ndim`, ALWAYS explicit (§3.2).
    pub physical_strides: [i64; 6],
    /// FDX logical dtype code of one pool element (the TRUE per-token type).
    pub element_dtype: u16,
    pub _pad3: [u8; 2],

    // ── BLOCK TABLE (logical → physical) ────────────────────────────────────
    pub block_table: FDXBlockTable,

    // ── LOGICAL (gathered) shape & liveness ─────────────────────────────────
    /// Batch size B (number of logical sequences).
    pub num_sequences: u64,
    /// Per-sequence logical CAPACITY = `max_blocks_per_seq * block_size` (P4).
    pub max_seq_capacity: u64,

    /// LOGICAL (gathered) per-sequence shape. `logical_ndim` gives the rank.
    pub logical_ndim: u8,
    /// Which axis of `logical_shape` is the per-sequence length. 0xFF if none.
    pub seq_axis: u8,
    pub _pad4: [u8; 6],
    pub logical_shape: [u64; 6],

    /// Per-LOGICAL-axis live extents, parallel to `logical_shape`.
    /// `logical_extents_count` is 0 or `logical_ndim` (V21e).
    pub logical_extents_count: u8,
    pub _pad5: [u8; 7],
    pub logical_extents: [FDXExtent; 6],

    // ── CONTEXT LENGTHS (per-sequence LIVE extent; symbolic — P4) ────────────
    /// Buffer-table index (§7.4) of the context_lens buffer (role =
    /// `FDX_BUFFER_CONTEXT_LENS`). `FDX_BUFFER_NONE` if purely symbolic.
    pub context_lens_buffer: u32,
    /// When all sequences share ONE symbolic live length, its `SymId`.
    /// `FDX_SYM_NONE` if per-seq lengths differ.
    pub context_len_sym: u32,
    /// Scope hint for the context length symbol (matches `FDXExtent.sym_scope`).
    pub context_len_scope: u8,
    pub _pad6: [u8; 3],

    pub reserved: [u32; 6],
}

/// A buffer reference in the side table (spec §7.2 / §7.4). All buffers a
/// logical tensor owns/references (data + scales + zero-points + bundle slots)
/// are collected and referenced by index; index 0 is always the base
/// `DLTensor.data` buffer (P7).
#[repr(C)]
pub struct FDXBufferRef {
    /// Role: 0 = Data, 1 = Scale, 2 = ZeroPoint, 3 = BundleBacking, 4 = Aux,
    /// 5 = `FDX_BUFFER_POOL`, 6 = `FDX_BUFFER_BLOCK_TABLE`,
    /// 7 = `FDX_BUFFER_CONTEXT_LENS` (§6.9.3).
    pub role: u8,
    pub _pad: [u8; 1],
    /// FDX logical dtype code of THIS buffer.
    pub dtype: u16,
    pub _pad2: u32,
    /// For the LIVE in-memory form only: device pointer (NEVER serialized). In
    /// serialized form this field is 0 and the byte location comes from the
    /// containing `DLManagedTensor` / file mapping.
    pub data: *mut c_void,
    pub device: DLDevice,
    /// Intra-buffer logical start (§3.3); MUST satisfy `byte_offset <= size_bytes`.
    pub byte_offset: u64,
    /// Physical allocated byte count of this buffer.
    pub size_bytes: u64,
    pub ndim: u32,
    pub _pad3: u32,
    pub shape: [u64; 6],
    /// Length `ndim`, ALWAYS explicit (§3.2); int64 — negatives first-class
    /// (§3.2.1 / V13). Never NULL.
    pub strides: [i64; 6],
    pub reserved: [u32; 4],
}

/// One `SymId → u64` binding in the per-pass call surface (spec §7.3). Never
/// serialized — per-call data, the sibling of the tensor-data cache.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FDXSymBinding {
    pub sym_id: u32,
    pub _pad: u32,
    /// `SymEnv` binds `usize`; widened to u64 (§6.4 narrowing policy).
    pub value: u64,
}

/// The boundary form of `fuel-core-types::symbol::SymEnv` (spec §7.3): a flat,
/// sorted array of `SymId → u64` bindings for O(log n) lookup. Never serialized.
#[repr(C)]
pub struct FDXSymEnv {
    pub count: u32,
    pub _pad: u32,
    /// Sorted by `sym_id`, write-once per pass.
    pub bindings: *const FDXSymBinding,
}

/// The top-level optional, versioned sidecar communicated ALONGSIDE a standard
/// `DLTensor` (spec §5.3). `#[repr(C)]` POD. Absence (a null `*const FDXSidecar`)
/// means "plain DLPack".
#[repr(C)]
pub struct FDXSidecar {
    /// `FDX_MAGIC`. Lets a consumer cheaply reject a misrouted pointer.
    pub magic: u32,
    /// `FDX_VERSION_*`. Major-only; INDEPENDENT of `DLPackVersion` (§5.2).
    pub version: u32,
    /// Total byte size of this struct as written by the producer. Enables
    /// size-prefixed forward-compat (P8). Feature detection is
    /// `(version, flags, struct_bytes)` jointly (§5.2, §14).
    pub struct_bytes: u32,
    /// `FDX_FLAG_*` bitmask.
    pub flags: u32,

    /// Sub-byte / microscaling dtype descriptor (§6.1). Valid iff
    /// `FDX_FLAG_HAS_DTYPE_EXT`.
    pub dtype_ext: FDXDTypeExt,

    /// Parametric quantization layout (§6.2). Valid iff `FDX_FLAG_HAS_QUANT`.
    pub quant: FDXQuant,

    /// Per-axis symbolic extents (§6.4). `extents_count == 0` ⇒ all axes are
    /// concrete; otherwise `extents_count == base.ndim`.
    pub extents_count: u32,
    pub _pad0: u32,
    pub extents: *const FDXExtent, // serialized: byte offset

    /// Tiling / alignment hints (§6.5). Valid iff `FDX_FLAG_HAS_TILING`.
    pub tiling: FDXTiling,

    /// Residency / substrate finer than `DLDevice` (§6.6).
    pub residency: FDXResidency,

    /// Storage class + session identity (§6.7).
    pub storage: FDXStorage,

    /// Side table of all buffers this logical tensor owns/references (§7.4).
    /// Index 0 is ALWAYS the base `DLTensor`'s data buffer.
    pub buffers_count: u32,
    pub _pad1: u32,
    pub buffers: *const FDXBufferRef, // serialized: byte offset

    /// Multi-output bundle sub-views (§6.8). Valid iff `FDX_FLAG_IS_BUNDLE`.
    pub views_count: u32,
    pub _pad2: u32,
    pub views: *const FDXOutputView, // serialized: byte offset

    /// Gather / indexed-residency descriptor (§6.9). Valid iff
    /// `FDX_FLAG_HAS_GATHER`. When unset, this is all-zero
    /// (`kind = FDX_GATHER_NONE`). Embedded by value; carved from the former
    /// `reserved[8]` tail (the §5.4 size test pins `sizeof(FDXSidecar)`).
    pub gather: FDXIndexedResidency,

    /// Reserved for additive growth without bumping `version`. Zero on write.
    /// Shrunk from `[u64; 8]` to keep `FDXSidecar`'s size class stable across the
    /// gather addition.
    pub reserved: [u64; 2],
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    // ── Spec-STATED sizes (§5.4): these MUST equal the spec's numbers. ───────
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn spec_stated_sizes() {
        // §6.4 / §6.4.1: "EXACTLY 16 bytes" and "sizeof == 8 + 8 + 4*16 == 80".
        assert_eq!(size_of::<FDXAffineTerm>(), 16, "spec §6.4: FDXAffineTerm == 16");
        assert_eq!(size_of::<FDXAffine>(), 80, "spec §6.4: FDXAffine == 80");
    }

    // ── Computed-by-implementation sizes (§5.4 requires a pin; the numeric
    //    value is NOT spec-stated and is reported back for cross-check). ──────
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn computed_sizes() {
        assert_eq!(size_of::<FDXDTypeExt>(), 16);
        assert_eq!(size_of::<FDXQuant>(), 100);
        assert_eq!(size_of::<FDXExtent>(), 128);
        assert_eq!(size_of::<FDXTiling>(), 48);
        assert_eq!(size_of::<FDXResidency>(), 32);
        assert_eq!(size_of::<FDXStorage>(), 32);
        assert_eq!(size_of::<FDXOutputView>(), 144);
        assert_eq!(size_of::<FDXBlockTable>(), 36);
        assert_eq!(size_of::<FDXIndexedResidency>(), 1064);
        assert_eq!(size_of::<FDXBufferRef>(), 160);
        assert_eq!(size_of::<FDXSymBinding>(), 16);
        assert_eq!(size_of::<FDXSymEnv>(), 16);
        assert_eq!(size_of::<FDXSidecar>(), 1376);
    }

    // ── FDXExtent field offsets — FROZEN per §6.4 layout rule. A future
    //    field-order edit breaks the build here, not silently at the ABI. ─────
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn fdx_extent_field_offsets_are_frozen() {
        assert_eq!(offset_of!(FDXExtent, kind), 0);
        assert_eq!(offset_of!(FDXExtent, _pad), 1);
        assert_eq!(offset_of!(FDXExtent, min), 8);
        assert_eq!(offset_of!(FDXExtent, capacity), 16);
        assert_eq!(offset_of!(FDXExtent, sym_id), 24);
        assert_eq!(offset_of!(FDXExtent, sym_scope), 28);
        assert_eq!(offset_of!(FDXExtent, _pad2), 29);
        assert_eq!(offset_of!(FDXExtent, cap_kind), 32);
        assert_eq!(offset_of!(FDXExtent, _pad3), 33);
        assert_eq!(offset_of!(FDXExtent, _pad4), 36);
        assert_eq!(offset_of!(FDXExtent, affine), 40);
        assert_eq!(offset_of!(FDXExtent, reserved), 120);
    }

    // ── FDXIndexedResidency trailing-field offsets — pinned per §5.4 so a
    //    future FDXExtent growth (which shifts everything after
    //    `logical_extents`) breaks the build HERE too, not at the ABI. ────────
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn fdx_indexed_residency_trailing_offsets_are_pinned() {
        assert_eq!(offset_of!(FDXIndexedResidency, logical_extents), 256);
        assert_eq!(offset_of!(FDXIndexedResidency, context_lens_buffer), 1024);
        assert_eq!(offset_of!(FDXIndexedResidency, context_len_sym), 1028);
        assert_eq!(offset_of!(FDXIndexedResidency, context_len_scope), 1032);
        assert_eq!(offset_of!(FDXIndexedResidency, reserved), 1036);
    }

    // ── FDXSidecar embeds `gather` by value and ends with reserved[u64; 2]
    //    (carved from the former reserved[8]); pin the gather offset so the
    //    carve-out stays put. ──────────────────────────────────────────────
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn fdx_sidecar_gather_is_embedded_by_value() {
        // gather + reserved[2] (16 bytes) must account for the tail.
        assert_eq!(
            offset_of!(FDXSidecar, reserved),
            offset_of!(FDXSidecar, gather) + size_of::<FDXIndexedResidency>()
        );
        assert_eq!(size_of::<FDXSidecar>(), offset_of!(FDXSidecar, reserved) + 16);
    }
}
