//! TDD for the FDX validators (V1..V21). For EVERY validator there is a test
//! that PASSES on a valid input and a test that FAILS with the EXPECTED
//! `FdxValidationError` variant on a crafted invalid input. Structs are built
//! directly. Tests run under `--features dlpack`.

use super::*;
use crate::dlpack::abi::{dtype_code, DLDataType, DLDevice};
use crate::dlpack::sidecar::{
    FDXAffine, FDXAffineTerm, FDXBlockTable, FDXBufferRef, FDXDTypeExt, FDXExtent,
    FDXIndexedResidency, FDXOutputView, FDXQuant, FDXResidency, FDXStorage, FDXSymBinding,
    FDXSymEnv, FDXTiling,
};
use core::ffi::c_void;

// ─────────────────────────────────────────────────────────────────────────────
// Builders
// ─────────────────────────────────────────────────────────────────────────────

fn uint8_dtype() -> DLDataType {
    DLDataType {
        code: dtype_code::K_DL_UINT,
        bits: 8,
        lanes: 1,
    }
}

fn f16_dtype() -> DLDataType {
    DLDataType {
        code: dtype_code::K_DL_FLOAT,
        bits: 16,
        lanes: 1,
    }
}

fn dtype_ext_none() -> FDXDTypeExt {
    FDXDTypeExt {
        logical_dtype: FDX_DTYPE_NONE,
        bit_width: 0,
        packing: 0,
        lanes: 0,
        sub_byte_bit_order: 0,
        _pad: 0,
        reserved: [0; 2],
    }
}

fn quant_none() -> FDXQuant {
    FDXQuant {
        family: FDX_QUANT_NONE,
        ggml_dtype: FDX_DTYPE_NONE,
        block_ndim: 0,
        _pad0: [0; 3],
        block_shape: [0; 4],
        block_axes: [-1; 4],
        pack_order: 0,
        _pad1: [0; 3],
        scale_present: 0,
        scale_dtype: FDX_DTYPE_NONE,
        scale_placement: 0,
        scale_granularity: 0,
        _pad2: [0; 3],
        scale_buffer: FDX_BUFFER_INLINE,
        zp_present: 0,
        zp_dtype: FDX_DTYPE_NONE,
        _pad3: 0,
        zp_buffer: FDX_BUFFER_INLINE,
        scale_pair_act: 0,
        scale_pair_weight: 0,
        role: 0,
        _pad4: 0,
        reserved: [0; 6],
    }
}

fn tiling_none() -> FDXTiling {
    FDXTiling {
        alignment_bytes: 0,
        access_granularity_bits: 0,
        tile_ndim: 0,
        _pad: [0; 7],
        tile_shape: [0; 4],
        reserved: [0; 4],
    }
}

fn residency_none() -> FDXResidency {
    FDXResidency {
        tier: 0,
        substrate: 0,
        backend_id: 0,
        _pad: 0,
        device_index: 0,
        is_mmap_view: 0,
        _pad2: [0; 7],
        reserved: [0; 4],
    }
}

fn storage_none() -> FDXStorage {
    FDXStorage {
        class: 0,
        _pad: [0; 3],
        _pad_align: 0,
        session_id: 0,
        reserved: [0; 4],
    }
}

fn gather_none() -> FDXIndexedResidency {
    FDXIndexedResidency {
        kind: FDX_GATHER_NONE as u8,
        _pad0: [0; 3],
        num_blocks: 0,
        block_size: 0,
        pool_buffer: 0,
        _pad1: 0,
        physical_ndim: 0,
        _pad2: [0; 7],
        physical_shape: [0; 6],
        physical_strides: [0; 6],
        element_dtype: FDX_DTYPE_NONE,
        _pad3: [0; 2],
        block_table: FDXBlockTable {
            table_buffer: 0,
            id_dtype: FDX_DTYPE_NONE,
            _pad0: 0,
            max_blocks_per_seq: 0,
            unmapped_sentinel: FDX_BLOCK_UNMAPPED,
            layout_flags: 0,
            reserved: [0; 4],
        },
        num_sequences: 0,
        max_seq_capacity: 0,
        logical_ndim: 0,
        seq_axis: 0xFF,
        _pad4: [0; 6],
        logical_shape: [0; 6],
        logical_extents_count: 0,
        _pad5: [0; 7],
        logical_extents: [scalar_ext(0); 6],
        context_lens_buffer: FDX_BUFFER_NONE,
        context_len_sym: FDX_SYM_NONE,
        context_len_scope: 0,
        _pad6: [0; 3],
        reserved: [0; 6],
    }
}

fn affine_zero() -> FDXAffine {
    FDXAffine {
        c0: 0,
        term_count: 0,
        _pad: [0; 7],
        terms: [FDXAffineTerm {
            coeff: 0,
            sym_id: FDX_SYM_NONE,
            _pad: 0,
        }; FDX_AFFINE_MAX_TERMS],
    }
}

fn scalar_ext(v: u64) -> FDXExtent {
    FDXExtent {
        kind: FDX_EXTENT_SCALAR as u8,
        _pad: [0; 3],
        min: v,
        capacity: v,
        sym_id: FDX_SYM_NONE,
        sym_scope: 0,
        _pad2: [0; 3],
        cap_kind: FDX_CAP_KIND_EXPLICIT as u8,
        _pad3: [0; 3],
        _pad4: 0,
        affine: affine_zero(),
        reserved: [0; 2],
    }
}

fn range_ext(min: u64, capacity: u64, sym_id: u32) -> FDXExtent {
    FDXExtent {
        kind: FDX_EXTENT_RANGE as u8,
        _pad: [0; 3],
        min,
        capacity,
        sym_id,
        sym_scope: 0,
        _pad2: [0; 3],
        cap_kind: FDX_CAP_KIND_EXPLICIT as u8,
        _pad3: [0; 3],
        _pad4: 0,
        affine: affine_zero(),
        reserved: [0; 2],
    }
}

fn affine_ext(min: u64, capacity: u64, c0: i64, terms: &[(i64, u32)]) -> FDXExtent {
    let mut a = affine_zero();
    a.c0 = c0;
    a.term_count = terms.len() as u8;
    for (i, &(coeff, sym_id)) in terms.iter().enumerate() {
        a.terms[i] = FDXAffineTerm {
            coeff,
            sym_id,
            _pad: 0,
        };
    }
    FDXExtent {
        kind: FDX_EXTENT_AFFINE as u8,
        _pad: [0; 3],
        min,
        capacity,
        sym_id: FDX_SYM_NONE,
        sym_scope: 0,
        _pad2: [0; 3],
        cap_kind: FDX_CAP_KIND_EXPLICIT as u8,
        _pad3: [0; 3],
        _pad4: 0,
        affine: a,
        reserved: [0; 2],
    }
}

fn data_buffer(size_bytes: u64) -> FDXBufferRef {
    FDXBufferRef {
        role: FDX_BUFFER_ROLE_DATA,
        _pad: [0; 1],
        dtype: FDX_DTYPE_U8,
        _pad2: 0,
        data: core::ptr::null_mut(),
        device: DLDevice {
            device_type: 1,
            device_id: 0,
        },
        byte_offset: 0,
        size_bytes,
        ndim: 1,
        _pad3: 0,
        shape: [size_bytes, 0, 0, 0, 0, 0],
        strides: [1, 0, 0, 0, 0, 0],
        reserved: [0; 4],
    }
}

/// A minimal valid sidecar for a meaning-bearing tensor (flags supplied).
fn sidecar(flags: u32) -> FDXSidecar {
    FDXSidecar {
        magic: FDX_MAGIC,
        version: FDX_VERSION_1,
        struct_bytes: core::mem::size_of::<FDXSidecar>() as u32,
        flags,
        dtype_ext: dtype_ext_none(),
        quant: quant_none(),
        extents_count: 0,
        _pad0: 0,
        extents: core::ptr::null(),
        tiling: tiling_none(),
        residency: residency_none(),
        storage: storage_none(),
        buffers_count: 0,
        _pad1: 0,
        buffers: core::ptr::null(),
        views_count: 0,
        _pad2: 0,
        views: core::ptr::null(),
        gather: gather_none(),
        reserved: [0; 2],
    }
}

/// Base honesty-uint8 1-D byte tensor of `n_bytes`.
fn base_uint8(n_bytes: i64, shape: &mut [i64; 1], strides: &mut [i64; 1]) -> DLTensor {
    shape[0] = n_bytes;
    strides[0] = 1;
    DLTensor {
        data: core::ptr::null_mut(),
        device: DLDevice {
            device_type: 1,
            device_id: 0,
        },
        ndim: 1,
        dtype: uint8_dtype(),
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn env(bindings: &[FDXSymBinding]) -> FDXSymEnv {
    FDXSymEnv {
        count: bindings.len() as u32,
        _pad: 0,
        bindings: bindings.as_ptr(),
    }
}

fn bind(sym_id: u32, value: u64) -> FDXSymBinding {
    FDXSymBinding {
        sym_id,
        _pad: 0,
        value,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V1 — header
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v1_pass() {
    let sc = sidecar(0);
    assert!(check_v1_header(&sc).is_ok());
}

#[test]
fn v1_fail_bad_magic() {
    let mut sc = sidecar(0);
    sc.magic = 0xDEAD_BEEF;
    assert!(matches!(
        check_v1_header(&sc),
        Err(FdxValidationError::BadMagic { .. })
    ));
}

#[test]
fn v1_fail_unsupported_version() {
    let mut sc = sidecar(0);
    sc.version = FDX_VERSION_MAX + 1;
    assert!(matches!(
        check_v1_header(&sc),
        Err(FdxValidationError::UnsupportedVersion { .. })
    ));
}

#[test]
fn v1_fail_struct_bytes_too_small() {
    let mut sc = sidecar(0);
    sc.struct_bytes = 4;
    assert!(matches!(
        check_v1_header(&sc),
        Err(FdxValidationError::StructBytesTooSmall { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V2 — flag/field coherence
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v2_pass() {
    let sc = sidecar(0); // no flags, no blocks
    assert!(check_v2_flag_coherence(&sc).is_ok());
}

#[test]
fn v2_fail_quant_flag_without_block() {
    let mut sc = sidecar(FDX_FLAG_HAS_QUANT); // flag set, quant.family == NONE
    sc.quant.family = FDX_QUANT_NONE;
    assert!(matches!(
        check_v2_flag_coherence(&sc),
        Err(FdxValidationError::FlagFieldIncoherent { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V3 — honesty (dtype)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v3_pass() {
    let sc = sidecar(FDX_FLAG_MEANING_REQUIRES_EXT);
    let mut sh = [0i64; 1];
    let mut st = [0i64; 1];
    let base = base_uint8(16, &mut sh, &mut st);
    assert!(check_v3_honesty_dtype(&sc, &base).is_ok());
}

#[test]
fn v3_fail_dishonest_dtype() {
    let sc = sidecar(FDX_FLAG_MEANING_REQUIRES_EXT);
    let mut sh = [4i64; 1];
    let mut st = [1i64; 1];
    let base = DLTensor {
        data: core::ptr::null_mut(),
        device: DLDevice {
            device_type: 1,
            device_id: 0,
        },
        ndim: 1,
        dtype: f16_dtype(), // dishonest: meaning-bearing but not uint8
        shape: sh.as_mut_ptr(),
        strides: st.as_mut_ptr(),
        byte_offset: 0,
    };
    assert!(matches!(
        check_v3_honesty_dtype(&sc, &base),
        Err(FdxValidationError::DishonestBase { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V4 — sub-byte sizing
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v4_pass() {
    let mut sc = sidecar(FDX_FLAG_HAS_DTYPE_EXT);
    sc.dtype_ext = FDXDTypeExt {
        logical_dtype: FDX_DTYPE_F4,
        bit_width: 4,
        packing: 1, // DENSE_SUBBYTE
        lanes: 1,
        sub_byte_bit_order: 0,
        _pad: 0,
        reserved: [0; 2],
    };
    assert!(check_v4_sub_byte(&sc).is_ok());
}

#[test]
fn v4_fail_zero_bit_width() {
    let mut sc = sidecar(FDX_FLAG_HAS_DTYPE_EXT);
    sc.dtype_ext.logical_dtype = FDX_DTYPE_F4;
    sc.dtype_ext.bit_width = 0;
    assert!(matches!(
        check_v4_sub_byte(&sc),
        Err(FdxValidationError::BadSubByte { .. })
    ));
}

#[test]
fn v4_fail_dense_subbyte_full_byte() {
    let mut sc = sidecar(FDX_FLAG_HAS_DTYPE_EXT);
    sc.dtype_ext.logical_dtype = FDX_DTYPE_U8;
    sc.dtype_ext.bit_width = 8;
    sc.dtype_ext.packing = 1; // DENSE_SUBBYTE but bit_width >= 8
    assert!(matches!(
        check_v4_sub_byte(&sc),
        Err(FdxValidationError::BadSubByte { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V5 — quant coherence
// ─────────────────────────────────────────────────────────────────────────────

fn ggml_quant() -> FDXQuant {
    let mut q = quant_none();
    q.family = FDX_QUANT_GGML_BLOCK;
    q.ggml_dtype = 12; // Q4K
    q.block_ndim = 1;
    q.block_shape = [256, 0, 0, 0];
    q.block_axes = [1, -1, -1, -1];
    q.scale_present = 1;
    q.scale_placement = FDX_SCALE_PLACEMENT_INLINE;
    q.scale_buffer = FDX_BUFFER_INLINE;
    q.scale_granularity = FDX_SCALE_GRAN_PER_TENSOR;
    q
}

fn mx_quant() -> FDXQuant {
    let mut q = quant_none();
    q.family = FDX_QUANT_MX;
    q.block_ndim = 1;
    q.block_shape = [32, 0, 0, 0];
    q.block_axes = [1, -1, -1, -1];
    q.scale_present = 1;
    q.scale_dtype = FDX_DTYPE_F8E8M0;
    q.scale_placement = FDX_SCALE_PLACEMENT_SEPARATE_BUFFER;
    q.scale_granularity = FDX_SCALE_GRAN_PER_BLOCK;
    q.scale_buffer = 1;
    q
}

fn affine_block_quant() -> FDXQuant {
    let mut q = quant_none();
    q.family = FDX_QUANT_AFFINE_BLOCK;
    q.block_ndim = 1;
    q.block_shape = [64, 0, 0, 0];
    q.block_axes = [0, -1, -1, -1];
    q.scale_present = 1;
    q.scale_dtype = FDX_DTYPE_F32;
    q.scale_placement = FDX_SCALE_PLACEMENT_SEPARATE_BUFFER;
    q.scale_buffer = 1;
    q.scale_granularity = FDX_SCALE_GRAN_PER_TENSOR;
    q
}

#[test]
fn v5_pass_ggml() {
    let mut sc = sidecar(FDX_FLAG_HAS_QUANT);
    sc.quant = ggml_quant();
    assert!(check_v5_quant(&sc).is_ok());
}

#[test]
fn v5_pass_mx() {
    let mut sc = sidecar(FDX_FLAG_HAS_QUANT);
    sc.quant = mx_quant();
    assert!(check_v5_quant(&sc).is_ok());
}

#[test]
fn v5_pass_affine_block() {
    let mut sc = sidecar(FDX_FLAG_HAS_QUANT);
    sc.quant = affine_block_quant();
    assert!(check_v5_quant(&sc).is_ok());
}

#[test]
fn v5_fail_ggml_with_perblock() {
    let mut sc = sidecar(FDX_FLAG_HAS_QUANT);
    let mut q = ggml_quant();
    q.scale_granularity = FDX_SCALE_GRAN_PER_BLOCK; // PerBlock under GGML
    sc.quant = q;
    assert!(matches!(
        check_v5_quant(&sc),
        Err(FdxValidationError::QuantRegimeViolation { .. })
    ));
}

#[test]
fn v5_fail_ggml_with_separate_buffer() {
    let mut sc = sidecar(FDX_FLAG_HAS_QUANT);
    let mut q = ggml_quant();
    q.scale_placement = FDX_SCALE_PLACEMENT_SEPARATE_BUFFER;
    q.scale_buffer = 1;
    sc.quant = q;
    assert!(matches!(
        check_v5_quant(&sc),
        Err(FdxValidationError::QuantRegimeViolation { .. })
    ));
}

#[test]
fn v5_fail_mx_wrong_scale_dtype() {
    let mut sc = sidecar(FDX_FLAG_HAS_QUANT);
    let mut q = mx_quant();
    q.scale_dtype = FDX_DTYPE_F32; // MX requires F8E8M0
    sc.quant = q;
    assert!(matches!(
        check_v5_quant(&sc),
        Err(FdxValidationError::QuantRegimeViolation { .. })
    ));
}

#[test]
fn v5_fail_affine_block_inline_scale() {
    let mut sc = sidecar(FDX_FLAG_HAS_QUANT);
    let mut q = affine_block_quant();
    q.scale_buffer = FDX_BUFFER_INLINE; // must be a real index
    sc.quant = q;
    assert!(matches!(
        check_v5_quant(&sc),
        Err(FdxValidationError::QuantRegimeViolation { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V6 — scale shape vs block geometry
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v6_pass_affine_block_scale_count() {
    // base logical axis 0 length 128, block 64 ⇒ 2 blocks ⇒ scale shape [2].
    let mut sc = sidecar(FDX_FLAG_HAS_QUANT);
    let mut q = affine_block_quant();
    q.block_axes = [0, -1, -1, -1];
    q.block_shape = [64, 0, 0, 0];
    sc.quant = q;
    let mut sh = [128i64; 1];
    let mut st = [1i64; 1];
    let base = base_uint8(128, &mut sh, &mut st);
    let mut scale = data_buffer(8);
    scale.role = FDX_BUFFER_ROLE_SCALE;
    scale.dtype = FDX_DTYPE_F32;
    scale.ndim = 1;
    scale.shape = [2, 0, 0, 0, 0, 0]; // 2 blocks
    let buffers = vec![data_buffer(64), scale];
    assert!(check_v6_scale_shape(&sc, &base, &buffers).is_ok());
}

#[test]
fn v6_affine_block_subbyte_logical_count() {
    // NF4 (spec §6.2 / §13.5a): the honest base is the PACKED uint8 byte buffer
    // (2 nibbles/byte), but the block count is over the base LOGICAL element shape.
    // 64 packed bytes @ bit_width 4 ⇒ 128 logical F4 elements; block 64 ⇒ 2 blocks
    // ⇒ scale shape [2]. V6 must convert the byte extent to logical via bit_width
    // (a byte-only `ceil(64/64)=1` would wrongly reject the spec's own example).
    let mut sc = sidecar(FDX_FLAG_HAS_QUANT | FDX_FLAG_HAS_DTYPE_EXT);
    sc.dtype_ext = FDXDTypeExt {
        logical_dtype: FDX_DTYPE_F4,
        bit_width: 4,
        packing: 1, // DENSE_SUBBYTE
        lanes: 1,
        sub_byte_bit_order: 0,
        _pad: 0,
        reserved: [0; 2],
    };
    let mut q = affine_block_quant();
    q.block_axes = [0, -1, -1, -1];
    q.block_shape = [64, 0, 0, 0];
    sc.quant = q;
    let mut sh = [64i64; 1]; // 64 PACKED bytes
    let mut st = [1i64; 1];
    let base = base_uint8(64, &mut sh, &mut st);
    let mut scale = data_buffer(8);
    scale.role = FDX_BUFFER_ROLE_SCALE;
    scale.dtype = FDX_DTYPE_F32;
    scale.ndim = 1;
    scale.shape = [2, 0, 0, 0, 0, 0]; // 128 logical / 64 = 2 blocks
    let buffers = vec![data_buffer(64), scale];
    assert!(
        check_v6_scale_shape(&sc, &base, &buffers).is_ok(),
        "sub-byte AFFINE_BLOCK block count must be over LOGICAL elements (bytes*8/bit_width)"
    );
}

#[test]
fn v6_fail_block_scale_count_mismatch() {
    let mut sc = sidecar(FDX_FLAG_HAS_QUANT);
    let mut q = affine_block_quant();
    q.block_axes = [0, -1, -1, -1];
    q.block_shape = [64, 0, 0, 0];
    sc.quant = q;
    let mut sh = [128i64; 1];
    let mut st = [1i64; 1];
    let base = base_uint8(128, &mut sh, &mut st);
    let mut scale = data_buffer(8);
    scale.role = FDX_BUFFER_ROLE_SCALE;
    scale.dtype = FDX_DTYPE_F32;
    scale.ndim = 1;
    scale.shape = [5, 0, 0, 0, 0, 0]; // wrong: should be 2
    let buffers = vec![data_buffer(64), scale];
    assert!(matches!(
        check_v6_scale_shape(&sc, &base, &buffers),
        Err(FdxValidationError::ScaleShapeMismatch { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V7 — extents
// ─────────────────────────────────────────────────────────────────────────────

fn base_3d(shape: &mut [i64; 3], strides: &mut [i64; 3]) -> DLTensor {
    // [32, 4096, 128] capacity F16, dense strides keyed to capacity.
    shape.copy_from_slice(&[32, 4096, 128]);
    strides.copy_from_slice(&[4096 * 128, 128, 1]);
    DLTensor {
        data: core::ptr::null_mut(),
        device: DLDevice {
            device_type: 2,
            device_id: 0,
        },
        ndim: 3,
        dtype: f16_dtype(),
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

#[test]
fn v7_pass_kv_extents() {
    let exts = [scalar_ext(32), range_ext(1, 4096, 7), scalar_ext(128)];
    let mut sc = sidecar(FDX_FLAG_HAS_SYMBOLIC);
    sc.extents_count = 3;
    sc.extents = exts.as_ptr();
    let mut sh = [0i64; 3];
    let mut st = [0i64; 3];
    let base = base_3d(&mut sh, &mut st);
    assert!(check_v7_extents(&sc, &base).is_ok());
}

#[test]
fn v7_fail_capacity_mismatch() {
    // axis 1 capacity 9999 != base.shape[1] 4096
    let exts = [scalar_ext(32), range_ext(1, 9999, 7), scalar_ext(128)];
    let mut sc = sidecar(FDX_FLAG_HAS_SYMBOLIC);
    sc.extents_count = 3;
    sc.extents = exts.as_ptr();
    let mut sh = [0i64; 3];
    let mut st = [0i64; 3];
    let base = base_3d(&mut sh, &mut st);
    assert!(matches!(
        check_v7_extents(&sc, &base),
        Err(FdxValidationError::ExtentMismatch { axis: 1, .. })
    ));
}

#[test]
fn v7_fail_cap_kind_poisoned() {
    let mut bad = scalar_ext(32);
    bad.cap_kind = 1; // AFFINE_MAX poisoning on a Scalar
    let exts = [bad, range_ext(1, 4096, 7), scalar_ext(128)];
    let mut sc = sidecar(FDX_FLAG_HAS_SYMBOLIC);
    sc.extents_count = 3;
    sc.extents = exts.as_ptr();
    let mut sh = [0i64; 3];
    let mut st = [0i64; 3];
    let base = base_3d(&mut sh, &mut st);
    assert!(matches!(
        check_v7_extents(&sc, &base),
        Err(FdxValidationError::ExtentMismatch { axis: 0, .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V8 — capacity backing
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v8_pass_full_backing() {
    let sc = sidecar(FDX_FLAG_HAS_SYMBOLIC);
    let mut sh = [0i64; 3];
    let mut st = [0i64; 3];
    let base = base_3d(&mut sh, &mut st);
    let buffers = vec![data_buffer(32 * 4096 * 128 * 2)];
    assert!(check_v8_capacity_backing(&sc, &base, &buffers).is_ok());
}

#[test]
fn v8_fail_under_backed_without_flag() {
    let sc = sidecar(FDX_FLAG_HAS_SYMBOLIC); // MEANING_REQUIRES_EXT clear
    let mut sh = [0i64; 3];
    let mut st = [0i64; 3];
    let base = base_3d(&mut sh, &mut st);
    let buffers = vec![data_buffer(1024)]; // far too small
    assert!(matches!(
        check_v8_capacity_backing(&sc, &base, &buffers),
        Err(FdxValidationError::CapacityNotBacked { .. })
    ));
}

#[test]
fn v8_pass_under_backed_with_meaning_flag() {
    let sc = sidecar(FDX_FLAG_HAS_SYMBOLIC | FDX_FLAG_MEANING_REQUIRES_EXT);
    let mut sh = [0i64; 3];
    let mut st = [0i64; 3];
    let base = base_3d(&mut sh, &mut st);
    let buffers = vec![data_buffer(1024)];
    assert!(check_v8_capacity_backing(&sc, &base, &buffers).is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// V9 — buffer refs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v9_pass() {
    let mut sc = sidecar(0);
    sc.buffers_count = 1;
    let buffers = vec![data_buffer(16)];
    assert!(check_v9_buffer_refs(&sc, &buffers).is_ok());
}

#[test]
fn v9_fail_scale_buffer_oob() {
    let mut sc = sidecar(FDX_FLAG_HAS_QUANT);
    sc.quant = mx_quant();
    sc.quant.scale_buffer = 7; // out of range
    sc.buffers_count = 1;
    let buffers = vec![data_buffer(16)];
    assert!(matches!(
        check_v9_buffer_refs(&sc, &buffers),
        Err(FdxValidationError::BufferRefOutOfRange { index: 7, .. })
    ));
}

#[test]
fn v9_fail_index0_not_data() {
    let mut sc = sidecar(0);
    sc.buffers_count = 1;
    let mut b = data_buffer(16);
    b.role = FDX_BUFFER_ROLE_SCALE; // index 0 must be Data
    let buffers = vec![b];
    assert!(matches!(
        check_v9_buffer_refs(&sc, &buffers),
        Err(FdxValidationError::FlagFieldIncoherent { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V10 — bundle
// ─────────────────────────────────────────────────────────────────────────────

fn view(byte_offset: u64, len_elements: u64, dtype: u16) -> FDXOutputView {
    FDXOutputView {
        byte_offset,
        len_elements,
        dtype,
        _pad: [0; 2],
        ndim: 1,
        shape: [len_elements, 0, 0, 0, 0, 0],
        strides: [1, 0, 0, 0, 0, 0],
        name_hash: 0,
        reserved: [0; 4],
    }
}

#[test]
fn v10_pass() {
    // bundle: F32 [0..B*V*4), I64 [B*V*4 ..]
    let b = 4u64;
    let v = 8u64;
    let total = b * v * 4 + b * 8;
    let views = [view(0, b * v, FDX_DTYPE_F32), view(b * v * 4, b, FDX_DTYPE_I64)];
    let mut sc = sidecar(FDX_FLAG_IS_BUNDLE);
    sc.views_count = 2;
    sc.views = views.as_ptr();
    let buffers = vec![data_buffer(total)];
    assert!(check_v10_bundle(&sc, &buffers).is_ok());
}

#[test]
fn v10_fail_overlap() {
    let views = [view(0, 8, FDX_DTYPE_F32), view(4, 8, FDX_DTYPE_F32)]; // overlap
    let mut sc = sidecar(FDX_FLAG_IS_BUNDLE);
    sc.views_count = 2;
    sc.views = views.as_ptr();
    let buffers = vec![data_buffer(1024)];
    assert!(matches!(
        check_v10_bundle(&sc, &buffers),
        Err(FdxValidationError::BundleOverlap { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V11 — explicit strides
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v11_pass() {
    let mut sh = [0i64; 1];
    let mut st = [0i64; 1];
    let base = base_uint8(16, &mut sh, &mut st);
    assert!(check_v11_explicit_strides(&base).is_ok());
}

#[test]
fn v11_fail_null_strides() {
    let mut sh = [4i64; 1];
    let base = DLTensor {
        data: core::ptr::null_mut(),
        device: DLDevice {
            device_type: 1,
            device_id: 0,
        },
        ndim: 1,
        dtype: f16_dtype(),
        shape: sh.as_mut_ptr(),
        strides: core::ptr::null_mut(), // NULL strides, ndim != 0
        byte_offset: 0,
    };
    assert!(matches!(
        check_v11_explicit_strides(&base),
        Err(FdxValidationError::NullStrides { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V12 — 256-byte alignment (boundary b)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v12_pass_aligned() {
    // 256-aligned pointer (fabricated; not dereferenced).
    let aligned = 0x1_0000usize as *mut c_void;
    let mut sh = [0i64; 1];
    let mut st = [0i64; 1];
    let mut base = base_uint8(16, &mut sh, &mut st);
    base.data = aligned;
    let mut buf = data_buffer(16);
    buf.data = aligned;
    let buffers = vec![buf];
    assert!(check_v12_alignment_boundary_b(&base, &buffers).is_ok());
}

#[test]
fn v12_fail_misaligned() {
    let misaligned = 0x1_0001usize as *mut c_void;
    let mut sh = [0i64; 1];
    let mut st = [0i64; 1];
    let mut base = base_uint8(16, &mut sh, &mut st);
    base.data = misaligned;
    let buffers = vec![data_buffer(16)];
    assert!(matches!(
        check_v12_alignment_boundary_b(&base, &buffers),
        Err(FdxValidationError::Misaligned { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V13 — signed-stride OOB range
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v13_pass_positive_strides() {
    // [4,4] F32, strides [4,1], size 64 bytes (4*4*4).
    assert!(check_v13_signed_stride_oob(&[4, 4], &[4, 1], 0, 4, 64, "t").is_ok());
}

#[test]
fn v13_pass_negative_strides_first_class() {
    // Reversed last axis: strides [4, -1], byte_offset points at iteration-first
    // element so start_offset stays non-negative. [4,4] F32 64 bytes.
    // For row r col c: addr = bo + r*4 + c*(-1). Window: lo at c=3 ⇒ -3;
    // byte_offset must absorb that. Put byte_offset at 3 elems = 12 bytes.
    assert!(check_v13_signed_stride_oob(&[4, 4], &[4, -1], 12, 4, 64, "t").is_ok());
}

#[test]
fn v13_fail_out_of_bounds() {
    // stride too large: [4] with stride 100, size 64 bytes ⇒ touches way past.
    assert!(matches!(
        check_v13_signed_stride_oob(&[4], &[100], 0, 4, 64, "t"),
        Err(FdxValidationError::StrideRangeOutOfBounds { .. })
    ));
}

#[test]
fn v13_fail_negative_escapes_below_zero() {
    // negative stride with insufficient byte_offset ⇒ window dips below 0.
    assert!(matches!(
        check_v13_signed_stride_oob(&[4], &[-1], 0, 4, 64, "t"),
        Err(FdxValidationError::StrideRangeOutOfBounds { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V14 — realize-time symbol bounds
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v14_pass_range_in_bounds() {
    let ext = range_ext(1, 4096, 7);
    let b = [bind(7, 2048)];
    let e = env(&b);
    assert_eq!(check_v14_extent_bounds(&ext, 1, &e).unwrap(), 2048);
}

#[test]
fn v14_fail_range_over_capacity() {
    let ext = range_ext(1, 4096, 7);
    let b = [bind(7, 5000)]; // > capacity
    let e = env(&b);
    assert!(matches!(
        check_v14_extent_bounds(&ext, 1, &e),
        Err(FdxValidationError::ExtentOutOfRange { .. })
    ));
}

#[test]
fn v14_fail_unbound_symbol() {
    let ext = range_ext(1, 4096, 7);
    let b: [FDXSymBinding; 0] = [];
    let e = env(&b);
    assert!(matches!(
        check_v14_extent_bounds(&ext, 1, &e),
        Err(FdxValidationError::UnboundSymbol { sym_id: 7, .. })
    ));
}

#[test]
fn v14_pass_affine_decode() {
    // k_len = cached_len(7) + new_tokens(8); cap 4096, min 1.
    let ext = affine_ext(1, 4096, 0, &[(1, 7), (1, 8)]);
    let b = [bind(7, 2000), bind(8, 1)];
    let e = env(&b);
    assert_eq!(check_v14_extent_bounds(&ext, 1, &e).unwrap(), 2001);
}

#[test]
fn v14_fail_affine_over_capacity() {
    let ext = affine_ext(1, 4096, 0, &[(1, 7), (1, 8)]);
    let b = [bind(7, 4096), bind(8, 1)]; // 4097 > 4096
    let e = env(&b);
    assert!(matches!(
        check_v14_extent_bounds(&ext, 1, &e),
        Err(FdxValidationError::ExtentOutOfRange { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V15 — no raw pointers in serialized form
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v15_pass_serialized() {
    let sc = sidecar(0); // all array pointers null
    let buffers = vec![data_buffer(16)]; // data null
    assert!(check_v15_no_serialized_pointers(&sc, &buffers).is_ok());
}

#[test]
fn v15_fail_pointer_present() {
    let mut sc = sidecar(0);
    let exts = [scalar_ext(1)];
    sc.extents = exts.as_ptr(); // non-null array pointer in "serialized" form
    sc.extents_count = 1;
    let buffers = vec![data_buffer(16)];
    assert!(matches!(
        check_v15_no_serialized_pointers(&sc, &buffers),
        Err(FdxValidationError::PointerInSerializedForm { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V16 — affine well-formedness
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v16_pass() {
    let ext = affine_ext(1, 4096, 0, &[(1, 7), (1, 8)]);
    assert!(check_v16_affine(&ext, 1).is_ok());
}

#[test]
fn v16_fail_degenerate_range() {
    // term_count==1, c0==0, coeff==1 ⇒ must be Range.
    let ext = affine_ext(0, 4096, 0, &[(1, 7)]);
    assert!(matches!(
        check_v16_affine(&ext, 1),
        Err(FdxValidationError::AffineDegenerate { .. })
    ));
}

#[test]
fn v16_fail_duplicate_sym() {
    let ext = affine_ext(1, 4096, 0, &[(1, 7), (1, 7)]); // dup sym 7
    assert!(matches!(
        check_v16_affine(&ext, 1),
        Err(FdxValidationError::AffineMalformed { .. })
    ));
}

#[test]
fn v16_fail_too_many_terms() {
    // craft term_count > MAX by hand.
    let mut ext = affine_ext(1, 4096, 0, &[(1, 1), (1, 2), (1, 3), (1, 4)]);
    ext.affine.term_count = (FDX_AFFINE_MAX_TERMS + 1) as u8;
    assert!(matches!(
        check_v16_affine(&ext, 1),
        Err(FdxValidationError::AffineTooManyTerms { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V17 — affine evaluation safety
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v17_pass() {
    let ext = affine_ext(1, 4096, 5, &[(1, 7)]); // c0=5 ⇒ legal affine
    let b = [bind(7, 100)];
    let e = env(&b);
    assert_eq!(check_v17_affine_eval(&ext, 1, &e).unwrap(), 105);
}

#[test]
fn v17_fail_overflow() {
    // coeff * sym overflows i128 accumulation.
    let ext = affine_ext(0, u64::MAX, 0, &[(i64::MAX, 7), (i64::MAX, 8)]);
    let b = [bind(7, u64::MAX), bind(8, u64::MAX)];
    let e = env(&b);
    assert!(matches!(
        check_v17_affine_eval(&ext, 1, &e),
        Err(FdxValidationError::AffineOverflow { .. })
    ));
}

#[test]
fn v17_fail_negative_result() {
    // c0 large-negative dominates ⇒ negative result rejected before narrowing.
    let ext = affine_ext(0, 4096, -1000, &[(1, 7)]);
    let b = [bind(7, 1)]; // -999
    let e = env(&b);
    assert!(matches!(
        check_v17_affine_eval(&ext, 1, &e),
        Err(FdxValidationError::ExtentOutOfRange { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// Gather builders
// ─────────────────────────────────────────────────────────────────────────────

/// A valid §13.8-style paged gather sidecar + base + buffers.
fn gather_fixture() -> (FDXSidecar, DLTensor, Vec<FDXBufferRef>, [i64; 1], [i64; 1]) {
    let num_blocks = 256u64;
    let block_size = 16u64;
    let hkv = 8u64;
    let d = 128u64;
    let elem = 2u64; // F16
    let per_block = block_size * hkv * d; // 16384 typed elements
    let pool_bytes = per_block * num_blocks * elem; // 8,388,608
    let max_blocks_per_seq = 64u32;
    let max_seq_cap = max_blocks_per_seq as u64 * block_size; // 1024
    let num_seq = 4u64;

    let mut g = gather_none();
    g.kind = FDX_GATHER_PAGED_BLOCKS as u8;
    g.num_blocks = num_blocks;
    g.block_size = block_size;
    g.pool_buffer = 0;
    g.physical_ndim = 4;
    g.physical_shape = [num_blocks, block_size, hkv, d, 0, 0];
    g.physical_strides = [(block_size * hkv * d) as i64, (hkv * d) as i64, d as i64, 1, 0, 0];
    g.element_dtype = FDX_DTYPE_F16;
    g.block_table = FDXBlockTable {
        table_buffer: 1,
        id_dtype: FDX_DTYPE_U32,
        _pad0: 0,
        max_blocks_per_seq,
        unmapped_sentinel: FDX_BLOCK_UNMAPPED,
        layout_flags: 0,
        reserved: [0; 4],
    };
    g.num_sequences = num_seq;
    g.max_seq_capacity = max_seq_cap;
    g.logical_ndim = 3;
    g.seq_axis = 1;
    g.logical_shape = [hkv, max_seq_cap, d, 0, 0, 0];
    g.logical_extents_count = 3;
    g.logical_extents = [
        scalar_ext(hkv),
        range_ext(0, max_seq_cap, 11),
        scalar_ext(d),
        scalar_ext(0),
        scalar_ext(0),
        scalar_ext(0),
    ];
    g.context_lens_buffer = 2;
    g.context_len_sym = 11;
    g.context_len_scope = 0;

    let mut sc = sidecar(FDX_FLAG_HAS_GATHER | FDX_FLAG_HAS_SYMBOLIC | FDX_FLAG_MEANING_REQUIRES_EXT);
    sc.gather = g;
    // base extents: 1-D byte pool, Scalar(pool_bytes).
    sc.buffers_count = 3;

    let mut sh = [0i64; 1];
    let mut st = [0i64; 1];
    let base = base_uint8(pool_bytes as i64, &mut sh, &mut st);

    let mut pool = data_buffer(pool_bytes);
    pool.role = FDX_BUFFER_ROLE_POOL;
    pool.dtype = FDX_DTYPE_F16;
    pool.ndim = 4;
    pool.shape = [num_blocks, block_size, hkv, d, 0, 0];
    pool.strides = [(block_size * hkv * d) as i64, (hkv * d) as i64, d as i64, 1, 0, 0];

    let mut bt = data_buffer(num_seq * max_blocks_per_seq as u64 * 4);
    bt.role = FDX_BUFFER_ROLE_BLOCK_TABLE;
    bt.dtype = FDX_DTYPE_U32;
    bt.ndim = 2;
    bt.shape = [num_seq, max_blocks_per_seq as u64, 0, 0, 0, 0];
    bt.strides = [max_blocks_per_seq as i64, 1, 0, 0, 0, 0];

    let mut cl = data_buffer(num_seq * 4);
    cl.role = FDX_BUFFER_ROLE_CONTEXT_LENS;
    cl.dtype = FDX_DTYPE_U32;
    cl.ndim = 1;
    cl.shape = [num_seq, 0, 0, 0, 0, 0];
    cl.strides = [1, 0, 0, 0, 0, 0];

    let buffers = vec![pool, bt, cl];
    (sc, base, buffers, sh, st)
}

// ─────────────────────────────────────────────────────────────────────────────
// V18 — gather coherence
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v18_pass() {
    let (sc, _b, _buf, _sh, _st) = gather_fixture();
    assert!(check_v18_gather_coherence(&sc).is_ok());
}

#[test]
fn v18_fail_block_size_zero() {
    let (mut sc, _b, _buf, _sh, _st) = gather_fixture();
    sc.gather.block_size = 0;
    assert!(matches!(
        check_v18_gather_coherence(&sc),
        Err(FdxValidationError::GatherIncoherent { .. })
    ));
}

#[test]
fn v18_fail_bad_capacity_product() {
    let (mut sc, _b, _buf, _sh, _st) = gather_fixture();
    sc.gather.max_seq_capacity = 999; // != 64*16
    assert!(matches!(
        check_v18_gather_coherence(&sc),
        Err(FdxValidationError::GatherIncoherent { .. })
    ));
}

#[test]
fn v18_fail_unsupported_kind() {
    let (mut sc, _b, _buf, _sh, _st) = gather_fixture();
    sc.gather.kind = 9; // unknown
    assert!(matches!(
        check_v18_gather_coherence(&sc),
        Err(FdxValidationError::UnsupportedGatherKind { kind: 9 })
    ));
}

#[test]
fn v18_fail_sentinel_below_num_blocks() {
    let (mut sc, _b, _buf, _sh, _st) = gather_fixture();
    sc.gather.block_table.unmapped_sentinel = 100; // < num_blocks 256
    assert!(matches!(
        check_v18_gather_coherence(&sc),
        Err(FdxValidationError::GatherIncoherent { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V19 — meaning + base honesty
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v19_pass() {
    let (sc, base, _buf, _sh, _st) = gather_fixture();
    assert!(check_v19_gather_meaning_and_honesty(&sc, &base).is_ok());
}

#[test]
fn v19_fail_missing_meaning_flag() {
    let (mut sc, base, _buf, _sh, _st) = gather_fixture();
    sc.flags &= !FDX_FLAG_MEANING_REQUIRES_EXT;
    assert!(matches!(
        check_v19_gather_meaning_and_honesty(&sc, &base),
        Err(FdxValidationError::DishonestBase { .. })
    ));
}

#[test]
fn v19_fail_base_byte_cover_mismatch() {
    let (sc, _base, _buf, mut sh, mut st) = gather_fixture();
    // base byte length wrong (half the pool).
    let base = base_uint8(4_194_304, &mut sh, &mut st);
    assert!(matches!(
        check_v19_gather_meaning_and_honesty(&sc, &base),
        Err(FdxValidationError::DishonestBase { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V20 — pool backing
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v20_pass() {
    let (sc, _b, buffers, _sh, _st) = gather_fixture();
    assert!(check_v20_pool_backing(&sc, &buffers).is_ok());
}

#[test]
fn v20_fail_pool_under_backed() {
    let (sc, _b, mut buffers, _sh, _st) = gather_fixture();
    buffers[0].size_bytes = 1024; // far too small
    assert!(matches!(
        check_v20_pool_backing(&sc, &buffers),
        Err(FdxValidationError::CapacityNotBacked { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V21(a) — gather buffers
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v21a_pass() {
    let (sc, _b, buffers, _sh, _st) = gather_fixture();
    assert!(check_v21a_gather_buffers(&sc, &buffers).is_ok());
}

#[test]
fn v21a_fail_block_table_shape() {
    let (sc, _b, mut buffers, _sh, _st) = gather_fixture();
    buffers[1].shape = [4, 99, 0, 0, 0, 0]; // wrong max_blocks_per_seq
    assert!(matches!(
        check_v21a_gather_buffers(&sc, &buffers),
        Err(FdxValidationError::GatherIncoherent { .. })
    ));
}

#[test]
fn v21a_fail_table_buffer_oob() {
    let (mut sc, _b, buffers, _sh, _st) = gather_fixture();
    sc.gather.block_table.table_buffer = 9; // out of range
    assert!(matches!(
        check_v21a_gather_buffers(&sc, &buffers),
        Err(FdxValidationError::BufferRefOutOfRange { index: 9, .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V21(c) — block table full-table scan
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v21c_pass() {
    let (sc, _b, _buf, _sh, _st) = gather_fixture();
    // valid ids plus sentinel tail.
    let ids = vec![0u32, 1, 2, 255, FDX_BLOCK_UNMAPPED, FDX_BLOCK_UNMAPPED];
    assert!(check_v21c_block_table_scan(&sc, &ids).is_ok());
}

#[test]
fn v21c_fail_id_out_of_range() {
    let (sc, _b, _buf, _sh, _st) = gather_fixture();
    let ids = vec![0u32, 256, 1]; // 256 >= num_blocks 256
    assert!(matches!(
        check_v21c_block_table_scan(&sc, &ids),
        Err(FdxValidationError::BlockIdOutOfRange { id: 256, .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V21(d) — seq live length
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v21d_pass_normal_and_empty() {
    let (sc, _b, _buf, _sh, _st) = gather_fixture();
    assert!(check_v21d_seq_live_length(&sc, 0, 512).is_ok());
    // L == 0 is legal (finished/evicted sequence).
    assert!(check_v21d_seq_live_length(&sc, 1, 0).is_ok());
    // exactly capacity is legal.
    assert!(check_v21d_seq_live_length(&sc, 2, 1024).is_ok());
}

#[test]
fn v21d_fail_over_capacity() {
    let (sc, _b, _buf, _sh, _st) = gather_fixture();
    assert!(matches!(
        check_v21d_seq_live_length(&sc, 0, 2000), // > max_seq_capacity 1024
        Err(FdxValidationError::ExtentOutOfRange { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// V21(e) — logical extents
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn v21e_pass() {
    let (sc, _b, _buf, _sh, _st) = gather_fixture();
    assert!(check_v21e_logical_extents(&sc).is_ok());
}

#[test]
fn v21e_fail_seq_extent_min_nonzero() {
    let (mut sc, _b, _buf, _sh, _st) = gather_fixture();
    sc.gather.logical_extents[1].min = 1; // must be 0 for a paged batch
    assert!(matches!(
        check_v21e_logical_extents(&sc),
        Err(FdxValidationError::GatherIncoherent { .. })
    ));
}

#[test]
fn v21e_fail_seq_sym_mismatch() {
    let (mut sc, _b, _buf, _sh, _st) = gather_fixture();
    sc.gather.logical_extents[1].sym_id = 99; // != context_len_sym 11
    assert!(matches!(
        check_v21e_logical_extents(&sc),
        Err(FdxValidationError::GatherIncoherent { .. })
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level surfaces
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn validate_pass_dense_kv() {
    let exts = [scalar_ext(32), range_ext(1, 4096, 7), scalar_ext(128)];
    let mut sc = sidecar(FDX_FLAG_HAS_SYMBOLIC);
    sc.extents_count = 3;
    sc.extents = exts.as_ptr();
    sc.buffers_count = 1;
    let mut sh = [0i64; 3];
    let mut st = [0i64; 3];
    let base = base_3d(&mut sh, &mut st);
    let buffers = vec![data_buffer(32 * 4096 * 128 * 2)];
    assert!(validate(&sc, &base, &buffers).is_ok());
}

#[test]
fn validate_pass_full_gather() {
    let (sc, base, buffers, _sh, _st) = gather_fixture();
    assert!(validate(&sc, &base, &buffers).is_ok());
}

#[test]
fn validate_realize_pass_gather() {
    let (sc, base, buffers, _sh, _st) = gather_fixture();
    let b = [bind(11, 512)];
    let e = env(&b);
    assert!(validate_realize(&sc, &base, &buffers, &e).is_ok());
}

#[test]
fn validate_realize_fail_over_capacity() {
    let (sc, base, buffers, _sh, _st) = gather_fixture();
    let b = [bind(11, 99999)]; // > max_seq_capacity
    let e = env(&b);
    assert!(validate_realize(&sc, &base, &buffers, &e).is_err());
}

#[test]
fn validate_realize_pass_affine_decode() {
    // [32, 4096, 128] KV with affine live axis 1.
    let exts = [
        scalar_ext(32),
        affine_ext(1, 4096, 0, &[(1, 7), (1, 8)]),
        scalar_ext(128),
    ];
    let mut sc = sidecar(FDX_FLAG_HAS_SYMBOLIC | FDX_FLAG_HAS_AFFINE_EXTENT);
    sc.extents_count = 3;
    sc.extents = exts.as_ptr();
    sc.buffers_count = 1;
    let mut sh = [0i64; 3];
    let mut st = [0i64; 3];
    let base = base_3d(&mut sh, &mut st);
    let buffers = vec![data_buffer(32 * 4096 * 128 * 2)];
    let b = [bind(7, 2000), bind(8, 1)];
    let e = env(&b);
    assert!(validate_realize(&sc, &base, &buffers, &e).is_ok());
}
