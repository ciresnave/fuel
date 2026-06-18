/*
 * fuel_dlpack_ext.h — the co-maintained C header for FDX (the Fuel DLPack
 * eXtension): a versioned, optional sidecar over standard DLPack for tensor
 * interchange between Fuel, its kernels (Baracuda CUDA / Vulkane Slang), and
 * the wider DLPack ecosystem.
 *
 * Canonical design: docs/specs/dlpack-extension.md (§5, §6, §7, Appendix A).
 *
 * ───────────────────────────────────────────────────────────────────────────
 * SOURCE OF TRUTH. The Rust `#[repr(C)]` definitions in
 *   fuel-core-types/src/dlpack/{abi.rs, sidecar.rs, codes.rs}
 * are now authoritative. This header mirrors them BYTE-FOR-BYTE on a 64-bit
 * little-endian host (the v1 target, §5): same field names, same field order,
 * same `_pad*`/`reserved` arrays. Every struct carries a `_Static_assert` on
 * its `sizeof` taken from the Rust `size_of` pins, plus `offsetof` asserts for
 * the load-bearing offsets the Rust pins. A Rust cross-check test
 * (fuel-core-types/src/dlpack/header_check.rs) `include_str!`s this file and
 * gates header↔Rust size agreement at `cargo test` — no C compiler needed.
 *
 * If you change a struct here, change the Rust (and vice versa); the cargo gate
 * will catch a size drift but NOT a same-size field re-shuffle, so keep field
 * order and names in lock-step by hand.
 * ───────────────────────────────────────────────────────────────────────────
 */
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C11 `_Static_assert` is required (also a C++11 keyword). */

/* ===========================================================================
 * STANDARD DLPACK (reproduced from dlpack.h v1.3 — FDX does NOT redefine
 * semantics; it consumes these unchanged). See abi.rs / spec §3, §5.1.
 * ===========================================================================
 */

/* DLPack standard flags carried on DLManagedTensorVersioned.flags (dlpack.h).
 * FDX relies on these directly and never redefines them. */
#define DLPACK_FLAG_BITMASK_READ_ONLY              (1UL << 0)
#define DLPACK_FLAG_BITMASK_IS_COPIED              (1UL << 1)
#define DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED (1UL << 2)

/* dlpack.h DLDevice. */
typedef struct {
  int32_t device_type;
  int32_t device_id;
} DLDevice;
_Static_assert(sizeof(DLDevice) == 8, "DLDevice size");

/* dlpack.h DLDataType — code/bits/lanes. */
typedef struct {
  uint8_t  code;
  uint8_t  bits;
  uint16_t lanes;
} DLDataType;
_Static_assert(sizeof(DLDataType) == 4, "DLDataType size");

/* dlpack.h DLTensor. `data` is 256-byte aligned on export (§3.3); the logical
 * start rides `byte_offset`. `strides` has length `ndim`, never NULL on a
 * versioned export (§3.2), and may be negative (§3.2.1). */
typedef struct {
  void*      data;
  DLDevice   device;
  int32_t    ndim;
  DLDataType dtype;
  int64_t*   shape;   /* length ndim; capacity bounds for symbolic axes */
  int64_t*   strides; /* length ndim, never NULL on a versioned export */
  uint64_t   byte_offset;
} DLTensor;
_Static_assert(sizeof(DLTensor) == 48, "DLTensor size");

/* dlpack.h DLPackVersion — the DLPack ABI version, independent of the FDX
 * schema version (§5.2). */
typedef struct {
  uint32_t major;
  uint32_t minor;
} DLPackVersion;
_Static_assert(sizeof(DLPackVersion) == 8, "DLPackVersion size");

/* dlpack.h DLManagedTensorVersioned — the cross-runtime managed form. At
 * boundary (b) the FDX sidecar rides `manager_ctx`, recovered only when the
 * live `deleter` identity matches Fuel's own (§10.2). The `deleter` and
 * `manager_ctx` pointers are never serialized. */
typedef struct DLManagedTensorVersioned {
  DLPackVersion version;
  void*         manager_ctx;
  void (*deleter)(struct DLManagedTensorVersioned* self);
  uint64_t      flags;
  DLTensor      dl_tensor;
} DLManagedTensorVersioned;
_Static_assert(sizeof(DLManagedTensorVersioned) == 80,
               "DLManagedTensorVersioned size");

/* ===========================================================================
 * FDX CONSTANTS — values EXACTLY match codes.rs (the normative owner).
 * ===========================================================================
 */

/* --- magic / schema version (§5.2) --- */
#define FDX_MAGIC       0x46445800u /* "FDX\0" */
#define FDX_VERSION_1   1u
#define FDX_VERSION_MAX 1u

/* --- sentinels (Appendix A) --- */
#define FDX_SYM_NONE       0xFFFFFFFFu
#define FDX_SESSION_NONE   0u
#define FDX_DTYPE_NONE     0xFFFFu
#define FDX_BUFFER_INLINE  0xFFFFFFFFu
#define FDX_BUFFER_NONE    0xFFFFFFFEu
#define FDX_BLOCK_UNMAPPED 0xFFFFFFFFu

/* --- FDXSidecar.flags (§5.2 authoritative bit-allocation table; u32) --- */
#define FDX_FLAG_HAS_DTYPE_EXT       (1u << 0)
#define FDX_FLAG_HAS_QUANT           (1u << 1)
#define FDX_FLAG_HAS_SYMBOLIC        (1u << 2)
#define FDX_FLAG_HAS_TILING          (1u << 3)
#define FDX_FLAG_IS_BUNDLE           (1u << 4)
#define FDX_FLAG_MEANING_REQUIRES_EXT (1u << 5)
#define FDX_FLAG_READ_ONLY           (1u << 6)
#define FDX_FLAG_HAS_GATHER          (1u << 7)
#define FDX_FLAG_HAS_AFFINE_EXTENT   (1u << 8)

/* --- FDXExtent.kind (§6.4) --- */
#define FDX_EXTENT_SCALAR 0u
#define FDX_EXTENT_RANGE  1u
#define FDX_EXTENT_AFFINE 2u
/* Affine term-count cap; overflow is a typed error (§6.4). */
#define FDX_AFFINE_MAX_TERMS 4u
/* FDXExtent.cap_kind (affine). */
#define FDX_CAP_KIND_EXPLICIT   0u
#define FDX_CAP_KIND_AFFINE_MAX 1u

/* --- FDXIndexedResidency.kind (gather, §6.9) --- */
#define FDX_GATHER_NONE         0u
#define FDX_GATHER_PAGED_BLOCKS 1u

/* --- FDXQuant.family (§6.2) — FDX is the normative owner --- */
#define FDX_QUANT_NONE         0xFFFFu
#define FDX_QUANT_GGML_BLOCK   0u
#define FDX_QUANT_MX           1u
#define FDX_QUANT_AFFINE_INT   2u
#define FDX_QUANT_AFFINE_FLOAT 3u
#define FDX_QUANT_AFFINE_BLOCK 4u

/* --- FDXScaleGranularity (§6.2) --- */
#define FDX_SCALE_GRAN_PER_TENSOR  0u
#define FDX_SCALE_GRAN_PER_TOKEN   1u
#define FDX_SCALE_GRAN_PER_CHANNEL 2u
#define FDX_SCALE_GRAN_PER_BLOCK   3u

/* --- FDXScalePlacement (§6.2) --- */
#define FDX_SCALE_PLACEMENT_INLINE           0u
#define FDX_SCALE_PLACEMENT_SEPARATE_BUFFER  1u
#define FDX_SCALE_PLACEMENT_BROADCAST_PER_AXIS 2u

/* --- FDXBufferRef.role (§7.2, §6.9.3) --- */
#define FDX_BUFFER_ROLE_DATA           0u
#define FDX_BUFFER_ROLE_SCALE          1u
#define FDX_BUFFER_ROLE_ZERO_POINT     2u
#define FDX_BUFFER_ROLE_BUNDLE_BACKING 3u
#define FDX_BUFFER_ROLE_AUX            4u
#define FDX_BUFFER_ROLE_POOL           5u
#define FDX_BUFFER_ROLE_BLOCK_TABLE    6u
#define FDX_BUFFER_ROLE_CONTEXT_LENS   7u

/* --- FDXResidency.tier (§6.6) — three-tier residency --- */
#define FDX_TIER_DEVICE    0u
#define FDX_TIER_HOST      1u
#define FDX_TIER_DISK_MMAP 2u

/* --- FDXResidency.substrate (§6.0/§6.6) — FDX-owned --- */
#define FDX_SUBSTRATE_HOST_BYTES    0u
#define FDX_SUBSTRATE_CUDA_UNTYPED  1u
#define FDX_SUBSTRATE_VULKAN_BUFFER 2u
#define FDX_SUBSTRATE_METAL_BUFFER  3u

/* --- FDXResidency.backend_id (§6.0/§6.6) — FDX-owned --- */
#define FDX_BACKEND_CPU    0u
#define FDX_BACKEND_CUDA   1u
#define FDX_BACKEND_VULKAN 2u
#define FDX_BACKEND_METAL  3u

/* --- FDXStorage.class (§6.7) --- */
#define FDX_STORAGE_SHARED    0u
#define FDX_STORAGE_SESSION   1u
#define FDX_STORAGE_TRANSIENT 2u

/* --- FDX ggml_dtype codes (§6.2) — FDX-owned; mirror GgmlDType::to_u32 --- */
#define FDX_GGML_F32  0u
#define FDX_GGML_F16  1u
#define FDX_GGML_Q4_0 2u
#define FDX_GGML_Q4_1 3u
#define FDX_GGML_Q5_0 6u
#define FDX_GGML_Q5_1 7u
#define FDX_GGML_Q8_0 8u
#define FDX_GGML_Q8_1 9u
#define FDX_GGML_Q2K  10u
#define FDX_GGML_Q3K  11u
#define FDX_GGML_Q4K  12u
#define FDX_GGML_Q5K  13u
#define FDX_GGML_Q6K  14u
#define FDX_GGML_Q8K  15u
#define FDX_GGML_BF16 30u

/* --- FDX logical dtype codes (§6.1 table) --- */
#define FDX_DTYPE_U8     0u
#define FDX_DTYPE_I8     1u
#define FDX_DTYPE_U32    2u
#define FDX_DTYPE_I16    3u
#define FDX_DTYPE_I32    4u
#define FDX_DTYPE_I64    5u
#define FDX_DTYPE_BF16   6u
#define FDX_DTYPE_F16    7u
#define FDX_DTYPE_F32    8u
#define FDX_DTYPE_F64    9u
#define FDX_DTYPE_F8E4M3 10u
#define FDX_DTYPE_F6E2M3 11u
#define FDX_DTYPE_F6E3M2 12u
#define FDX_DTYPE_F4     13u
#define FDX_DTYPE_F8E8M0 14u

/* ===========================================================================
 * FDX STRUCTS — field-for-field mirror of sidecar.rs. Layout matches the Rust
 * on a 64-bit LE host. Pointer fields marked "live only" are never serialized
 * (the serialized form replaces a pointer with a byte offset; P7).
 * ===========================================================================
 */

/* Sub-byte / microscaling dtype descriptor (§6.1). Valid iff
 * FDX_FLAG_HAS_DTYPE_EXT. */
typedef struct {
  uint16_t logical_dtype;
  uint16_t bit_width;
  uint8_t  packing;
  uint8_t  lanes;
  uint8_t  sub_byte_bit_order;
  uint8_t  _pad;
  uint32_t reserved[2];
} FDXDTypeExt;
_Static_assert(sizeof(FDXDTypeExt) == 16, "FDXDTypeExt size");

/* Parametric quantization block layout (§6.2). Valid iff FDX_FLAG_HAS_QUANT. */
typedef struct {
  uint16_t family;
  uint16_t ggml_dtype;

  uint8_t  block_ndim;
  uint8_t  _pad0[3];
  uint32_t block_shape[4];
  int32_t  block_axes[4];

  uint8_t  pack_order;
  uint8_t  _pad1[3];

  uint8_t  scale_present;
  uint16_t scale_dtype;
  uint8_t  scale_placement;
  uint8_t  scale_granularity;
  uint8_t  _pad2[3];
  uint32_t scale_buffer;

  uint8_t  zp_present;
  uint16_t zp_dtype;
  uint8_t  _pad3;
  uint32_t zp_buffer;

  uint8_t  scale_pair_act;
  uint8_t  scale_pair_weight;
  uint8_t  role;
  uint8_t  _pad4;

  uint32_t reserved[6];
} FDXQuant;
_Static_assert(sizeof(FDXQuant) == 100, "FDXQuant size");

/* One affine term `coeff * sym_id` (§6.4). EXACTLY 16 bytes. */
typedef struct {
  int64_t  coeff;
  uint32_t sym_id;
  uint32_t _pad;
} FDXAffineTerm;
_Static_assert(sizeof(FDXAffineTerm) == 16, "FDXAffineTerm size");

/* value = c0 + Σ_{i<term_count} terms[i].coeff * resolve(terms[i].sym_id)
 * (§6.4). Fixed-capacity (FDX_AFFINE_MAX_TERMS = 4), inline, POD. EXACTLY 80
 * bytes. */
typedef struct {
  int64_t       c0;
  uint8_t       term_count;
  uint8_t       _pad[7];
  FDXAffineTerm terms[4]; /* FDX_AFFINE_MAX_TERMS */
} FDXAffine;
_Static_assert(sizeof(FDXAffine) == 80, "FDXAffine size");

/* Symbolic / dynamic extent — live-vs-capacity (§6.4). Leading-byte layout is
 * FROZEN: kind@0, min@8, capacity@16, sym_id@24, sym_scope@28; affine machinery
 * appended strictly after offset 32. Doubly load-bearing (extents[] stride AND
 * FDXIndexedResidency.logical_extents[6] inline member). */
typedef struct {
  uint8_t   kind;
  uint8_t   _pad[3];
  uint64_t  min;
  uint64_t  capacity;
  uint32_t  sym_id;
  uint8_t   sym_scope;
  uint8_t   _pad2[3];
  uint8_t   cap_kind;
  uint8_t   _pad3[3];
  uint32_t  _pad4;
  FDXAffine affine;
  uint32_t  reserved[2];
} FDXExtent;
_Static_assert(sizeof(FDXExtent) == 128, "FDXExtent size");
_Static_assert(offsetof(FDXExtent, kind) == 0, "FDXExtent.kind offset");
_Static_assert(offsetof(FDXExtent, _pad) == 1, "FDXExtent._pad offset");
_Static_assert(offsetof(FDXExtent, min) == 8, "FDXExtent.min offset");
_Static_assert(offsetof(FDXExtent, capacity) == 16, "FDXExtent.capacity offset");
_Static_assert(offsetof(FDXExtent, sym_id) == 24, "FDXExtent.sym_id offset");
_Static_assert(offsetof(FDXExtent, sym_scope) == 28, "FDXExtent.sym_scope offset");
_Static_assert(offsetof(FDXExtent, _pad2) == 29, "FDXExtent._pad2 offset");
_Static_assert(offsetof(FDXExtent, cap_kind) == 32, "FDXExtent.cap_kind offset");
_Static_assert(offsetof(FDXExtent, _pad3) == 33, "FDXExtent._pad3 offset");
_Static_assert(offsetof(FDXExtent, _pad4) == 36, "FDXExtent._pad4 offset");
_Static_assert(offsetof(FDXExtent, affine) == 40, "FDXExtent.affine offset");
_Static_assert(offsetof(FDXExtent, reserved) == 120, "FDXExtent.reserved offset");

/* Tiling / alignment hints (§6.5). Valid iff FDX_FLAG_HAS_TILING. */
typedef struct {
  uint32_t alignment_bytes;
  uint32_t access_granularity_bits;
  uint8_t  tile_ndim;
  uint8_t  _pad[7];
  uint32_t tile_shape[4];
  uint32_t reserved[4];
} FDXTiling;
_Static_assert(sizeof(FDXTiling) == 48, "FDXTiling size");

/* Residency / substrate, finer than DLDevice (§6.6). */
typedef struct {
  uint8_t  tier;
  uint8_t  substrate;
  uint8_t  backend_id;
  uint8_t  _pad;
  uint32_t device_index;
  uint8_t  is_mmap_view;
  uint8_t  _pad2[7];
  uint32_t reserved[4];
} FDXResidency;
_Static_assert(sizeof(FDXResidency) == 32, "FDXResidency size");

/* Storage class + session identity (§6.7). */
typedef struct {
  uint8_t  class_; /* Rust `class`; `class` is a C++ keyword, renamed here */
  uint8_t  _pad[3];
  uint32_t _pad_align;
  uint64_t session_id;
  uint32_t reserved[4];
} FDXStorage;
_Static_assert(sizeof(FDXStorage) == 32, "FDXStorage size");

/* Multi-output bundle sub-view (§6.8). Valid iff FDX_FLAG_IS_BUNDLE. */
typedef struct {
  uint64_t byte_offset;
  uint64_t len_elements;
  uint16_t dtype;
  uint8_t  _pad[2];
  uint32_t ndim;
  uint64_t shape[6];
  int64_t  strides[6];
  uint64_t name_hash;
  uint32_t reserved[4];
} FDXOutputView;
_Static_assert(sizeof(FDXOutputView) == 144, "FDXOutputView size");

/* Logical → physical block mapping for a paged pool (§6.9.2). */
typedef struct {
  uint32_t table_buffer;
  uint16_t id_dtype;
  uint16_t _pad0;
  uint32_t max_blocks_per_seq;
  uint32_t unmapped_sentinel;
  uint32_t layout_flags;
  uint32_t reserved[4];
} FDXBlockTable;
_Static_assert(sizeof(FDXBlockTable) == 36, "FDXBlockTable size");

/* GATHER / INDEXED-RESIDENCY descriptor (§6.9.2): re-interprets a contiguous
 * physical BLOCK POOL as a logically-gathered tensor via a per-sequence block
 * table. Embedded by value in FDXSidecar, valid iff FDX_FLAG_HAS_GATHER. */
typedef struct {
  uint8_t       kind;
  uint8_t       _pad0[3];

  uint64_t      num_blocks;
  uint64_t      block_size;

  uint32_t      pool_buffer;
  uint32_t      _pad1;

  uint8_t       physical_ndim;
  uint8_t       _pad2[7];
  uint64_t      physical_shape[6];
  int64_t       physical_strides[6];
  uint16_t      element_dtype;
  uint8_t       _pad3[2];

  FDXBlockTable block_table;

  uint64_t      num_sequences;
  uint64_t      max_seq_capacity;

  uint8_t       logical_ndim;
  uint8_t       seq_axis;
  uint8_t       _pad4[6];
  uint64_t      logical_shape[6];

  uint8_t       logical_extents_count;
  uint8_t       _pad5[7];
  FDXExtent     logical_extents[6];

  uint32_t      context_lens_buffer;
  uint32_t      context_len_sym;
  uint8_t       context_len_scope;
  uint8_t       _pad6[3];

  uint32_t      reserved[6];
} FDXIndexedResidency;
_Static_assert(sizeof(FDXIndexedResidency) == 1064, "FDXIndexedResidency size");
_Static_assert(offsetof(FDXIndexedResidency, logical_extents) == 256,
               "FDXIndexedResidency.logical_extents offset");
_Static_assert(offsetof(FDXIndexedResidency, context_lens_buffer) == 1024,
               "FDXIndexedResidency.context_lens_buffer offset");
_Static_assert(offsetof(FDXIndexedResidency, context_len_sym) == 1028,
               "FDXIndexedResidency.context_len_sym offset");
_Static_assert(offsetof(FDXIndexedResidency, context_len_scope) == 1032,
               "FDXIndexedResidency.context_len_scope offset");
_Static_assert(offsetof(FDXIndexedResidency, reserved) == 1036,
               "FDXIndexedResidency.reserved offset");

/* A buffer reference in the side table (§7.2 / §7.4). Index 0 is always the
 * base DLTensor.data buffer (P7). */
typedef struct {
  uint8_t  role;
  uint8_t  _pad[1];
  uint16_t dtype;
  uint32_t _pad2;
  void*    data; /* live only: device pointer, NEVER serialized */
  DLDevice device;
  uint64_t byte_offset;
  uint64_t size_bytes;
  uint32_t ndim;
  uint32_t _pad3;
  uint64_t shape[6];
  int64_t  strides[6];
  uint32_t reserved[4];
} FDXBufferRef;
_Static_assert(sizeof(FDXBufferRef) == 160, "FDXBufferRef size");

/* One SymId → u64 binding in the per-pass call surface (§7.3). Never
 * serialized. */
typedef struct {
  uint32_t sym_id;
  uint32_t _pad;
  uint64_t value;
} FDXSymBinding;
_Static_assert(sizeof(FDXSymBinding) == 16, "FDXSymBinding size");

/* The boundary form of SymEnv (§7.3): a flat, sorted array of SymId → u64
 * bindings. Never serialized. */
typedef struct {
  uint32_t             count;
  uint32_t             _pad;
  const FDXSymBinding* bindings; /* live only: never serialized */
} FDXSymEnv;
_Static_assert(sizeof(FDXSymEnv) == 16, "FDXSymEnv size");

/* The top-level optional, versioned sidecar communicated ALONGSIDE a standard
 * DLTensor (§5.3). Absence (a null FDXSidecar*) means "plain DLPack". */
typedef struct {
  uint32_t            magic;
  uint32_t            version;
  uint32_t            struct_bytes;
  uint32_t            flags;

  FDXDTypeExt         dtype_ext;

  FDXQuant            quant;

  uint32_t            extents_count;
  uint32_t            _pad0;
  const FDXExtent*    extents; /* serialized: byte offset */

  FDXTiling           tiling;

  FDXResidency        residency;

  FDXStorage          storage;

  uint32_t            buffers_count;
  uint32_t            _pad1;
  const FDXBufferRef* buffers; /* serialized: byte offset */

  uint32_t            views_count;
  uint32_t            _pad2;
  const FDXOutputView* views; /* serialized: byte offset */

  FDXIndexedResidency gather;

  uint64_t            reserved[2];
} FDXSidecar;
_Static_assert(sizeof(FDXSidecar) == 1376, "FDXSidecar size");
_Static_assert(offsetof(FDXSidecar, gather) == 296, "FDXSidecar.gather offset");
_Static_assert(offsetof(FDXSidecar, reserved) == 1360,
               "FDXSidecar.reserved offset");
/* gather is embedded by value: reserved follows gather immediately. */
_Static_assert(offsetof(FDXSidecar, reserved)
                 == offsetof(FDXSidecar, gather) + sizeof(FDXIndexedResidency),
               "FDXSidecar gather embedded-by-value");

#ifdef __cplusplus
} /* extern "C" */
#endif
