//! Born-red tests for the DLPack + FDX borrowed view (plan §2, gates T7–T9).
//!
//! All CPU — the only backend wired in this slice. Each test asserts the
//! field-by-field mapping (§2.2) and the sidecar-presence rules (§2.3), and runs
//! the FDX validator surface so the honesty invariant is mechanically checked.

use super::*;
use fuel_core_types::dlpack::abi::{device_type, dtype_code};
use fuel_core_types::shape::Shape;
use fuel_core_types::symbol::SymId;
use fuel_core_types::{DType, Layout, SymEnv};
use fuel_cpu_backend::CpuStorageBytes;

use crate::{BackendStorage, Storage};

/// Build a CPU F32 `Storage` holding `n` zeroed elements.
fn cpu_f32(n: usize) -> Storage {
    Storage::new(
        BackendStorage::Cpu(CpuStorageBytes::from_zero_bytes(n * 4)),
        DType::F32,
    )
}

/// Build a CPU `Storage` of `len_bytes` zeroed bytes tagged with `dtype`.
fn cpu_bytes(dtype: DType, len_bytes: usize) -> Storage {
    Storage::new(
        BackendStorage::Cpu(CpuStorageBytes::from_zero_bytes(len_bytes)),
        dtype,
    )
}

/// The CPU backend's base byte pointer for a `Storage` (the same pointer
/// `view()` puts in `dl.data`). Robust across feature sets.
fn cpu_base_ptr(storage: &Storage) -> usize {
    #[allow(irrefutable_let_patterns)]
    if let BackendStorage::Cpu(c) = &storage.inner {
        c.bytes().as_ptr() as usize
    } else {
        panic!("test storage must be CPU")
    }
}

// ── T7a: plain contiguous F32 → faithful dtype, sidecar None ────────────────

#[test]
fn plain_f32_contiguous_is_faithful_no_sidecar() {
    let storage = cpu_f32(3 * 4); // [3,4]
    let layout = Layout::contiguous(Shape::from_dims(&[3, 4]));
    let v = view(&storage, &layout, None).expect("view");

    // sidecar absent ⇒ plain DLPack (P2).
    assert!(v.sidecar.is_none(), "plain F32 must have no sidecar");

    // faithful dtype {kDLFloat, 32, 1}.
    assert_eq!(v.dl.dtype.code, dtype_code::K_DL_FLOAT);
    assert_eq!(v.dl.dtype.bits, 32);
    assert_eq!(v.dl.dtype.lanes, 1);

    // ndim/shape == layout.dims() (capacity).
    assert_eq!(v.ndim(), 2);
    assert_eq!(v.shape(), &[3, 4]);

    // row-major strides, explicit (never NULL).
    assert_eq!(v.strides(), &[4, 1]);

    // byte_offset == start_offset * size_in_bytes (0 here).
    assert_eq!(v.dl.byte_offset, 0);

    // dl_tensor() materializes valid shape/strides pointers.
    let dl = v.dl_tensor();
    assert!(!dl.shape.is_null());
    assert!(!dl.strides.is_null());
    assert_eq!(unsafe { core::slice::from_raw_parts(dl.shape, 2) }, &[3, 4]);
    assert_eq!(unsafe { core::slice::from_raw_parts(dl.strides, 2) }, &[4, 1]);

    // validator surface is a no-op (and Ok) for sidecar-None.
    v.validate().expect("plain view validates");
}

#[test]
fn plain_f32_with_offset_byte_offset_scales() {
    let storage = cpu_f32(100);
    // start_offset 5 elements → 20 bytes.
    let layout = Layout::contiguous_with_offset(Shape::from_dims(&[4]), 5);
    let v = view(&storage, &layout, None).expect("view");
    assert_eq!(v.dl.byte_offset, 20);
    assert_eq!(v.strides(), &[1]);
}

// ── T7b: device pointer is the borrowed base, no offset folded ──────────────

#[test]
fn data_pointer_is_storage_base_and_cpu_device() {
    let storage = cpu_f32(8);
    let layout = Layout::contiguous_with_offset(Shape::from_dims(&[2]), 3);
    let v = view(&storage, &layout, None).expect("view");

    // dl.data is the base pointer (NOT offset-folded — the offset is in
    // byte_offset).
    let expected = cpu_base_ptr(&storage);
    assert_eq!(v.dl.data as usize, expected);
    assert_eq!(v.dl.byte_offset, 12); // 3 * 4
    assert_eq!(v.dl.device.device_type, device_type::K_DL_CPU); // kDLCPU == 1
    assert_eq!(v.dl.device.device_id, 0);
}

// ── Negative-stride (Op::Flip) view: strides pass through unchanged ─────────

#[test]
fn flipped_layout_carries_negative_stride_and_iteration_first_offset() {
    // [3,4] contiguous, flip dim 0 → stride [-4, 1], start_offset 8.
    let storage = cpu_f32(3 * 4);
    let base = Layout::contiguous(Shape::from_dims(&[3, 4]));
    let flipped = base.flip(0).expect("flip");
    assert_eq!(flipped.stride(), &[-4_isize, 1]);
    assert_eq!(flipped.start_offset(), 8); // (3-1)*4 — the Layout invariant.

    let v = view(&storage, &flipped, None).expect("view");
    // Negative stride passes through unchanged.
    assert_eq!(v.strides(), &[-4, 1]);
    // byte_offset is the (non-negative) iteration-first offset, in bytes.
    assert_eq!(v.dl.byte_offset, 8 * 4); // 8 elements * 4 bytes = 32

    // Still faithful F32, no sidecar (a flip is orthogonal to FDX meaning).
    assert!(v.sidecar.is_none());
    assert_eq!(v.dl.dtype.code, dtype_code::K_DL_FLOAT);
}

// ── T7c: sub-byte F4 → uint8 stand-in + HAS_DTYPE_EXT + bit_width ───────────

#[test]
fn sub_byte_f4_uses_uint8_standin_with_dtype_ext() {
    // F4: 2 elements per byte. Logical [8, 4096] → 8*4096/2 = 16384 bytes.
    let len_bytes = 8 * 4096 / 2;
    let storage = cpu_bytes(DType::F4, len_bytes);
    let layout = Layout::contiguous(Shape::from_dims(&[8, 4096]));
    let v = view(&storage, &layout, None).expect("view");

    // Base dtype is the {kDLUInt, 8, 1} honesty stand-in.
    assert_eq!(v.dl.dtype.code, dtype_code::K_DL_UINT);
    assert_eq!(v.dl.dtype.bits, 8);
    assert_eq!(v.dl.dtype.lanes, 1);

    // Base is the honest 1-D physical byte buffer.
    assert_eq!(v.ndim(), 1);
    assert_eq!(v.shape(), &[len_bytes as i64]);
    assert_eq!(v.strides(), &[1]);

    // Sidecar present with HAS_DTYPE_EXT + a non-zero bit_width.
    let sc = v.sidecar.as_ref().expect("sub-byte sidecar");
    assert_ne!(sc.flags & FDX_FLAG_HAS_DTYPE_EXT, 0);
    assert_ne!(sc.flags & FDX_FLAG_MEANING_REQUIRES_EXT, 0);
    assert_eq!(sc.dtype_ext.logical_dtype, FDX_DTYPE_F4);
    assert_eq!(sc.dtype_ext.bit_width, 4);
    assert_ne!(sc.dtype_ext.bit_width, 0);
    assert_eq!(sc.dtype_ext.packing, 1); // DENSE_SUBBYTE

    // Validator confirms the honesty invariant mechanically (V3 dtype, V4
    // sub-byte sizing, V11 strides, V13 range).
    v.validate().expect("sub-byte view validates");
}

#[test]
fn sub_byte_f6_has_six_bit_width() {
    let storage = cpu_bytes(DType::F6E2M3, 96);
    let layout = Layout::contiguous(Shape::from_dims(&[128]));
    let v = view(&storage, &layout, None).expect("view");
    let sc = v.sidecar.as_ref().expect("sidecar");
    assert_eq!(sc.dtype_ext.bit_width, 6);
    assert_eq!(sc.dtype_ext.logical_dtype, FDX_DTYPE_F6E2M3);
}

// ── T8: symbolic layout → HAS_SYMBOLIC + extents; V14 in/out of range ───────

/// A KV-cache-style layout: `[n_heads=2, K_capacity=8, head_dim=4]` with the
/// middle axis bounded-symbolic `[min=1, max=8]` under `SymId(7)`.
fn symbolic_kv_layout(sym: SymId) -> Layout {
    let shape = Shape::from_dims(&[2, 8, 4]).with_dynamic_axis(1, 1, sym);
    Layout::contiguous(shape)
}

#[test]
fn symbolic_layout_transports_symbol_not_value() {
    let sym = SymId(7);
    // Allocate at full capacity (2*8*4 = 64 F32 elements) so V8 backing holds.
    let storage = cpu_f32(64);
    let layout = symbolic_kv_layout(sym);
    assert!(layout.has_dynamic());

    // No env: build-time only. Symbol transported, value not resolved.
    let v = view(&storage, &layout, None).expect("view");
    let sc = v.sidecar.as_ref().expect("symbolic sidecar");
    assert_ne!(sc.flags & FDX_FLAG_HAS_SYMBOLIC, 0);
    assert_eq!(sc.extents_count, 3);

    // Faithful F32 base (symbolic is orthogonal to dtype honesty).
    assert_eq!(v.dl.dtype.code, dtype_code::K_DL_FLOAT);
    // Capacity shape (max), strides keyed to capacity.
    assert_eq!(v.shape(), &[2, 8, 4]);
    assert_eq!(v.strides(), &[32, 4, 1]);

    // The middle extent carries the SymId and the [min, capacity] window — the
    // SYMBOL, never a resolved value (P4).
    let extents = v.sidecar.as_ref().unwrap();
    let ext = unsafe { &*extents.extents.add(1) };
    assert_eq!(ext.kind as u16, FDX_EXTENT_RANGE);
    assert_eq!(ext.sym_id, 7);
    assert_eq!(ext.min, 1);
    assert_eq!(ext.capacity, 8);

    // Build-time validate passes.
    v.validate().expect("symbolic build-time validates");
}

#[test]
fn symbolic_v14_passes_in_range_fails_out_of_range() {
    let sym = SymId(7);
    let storage = cpu_f32(64);
    let layout = symbolic_kv_layout(sym);

    // In-range: live = 5 ∈ [1, 8]. With a binding env, view() runs V14 and
    // succeeds.
    let mut env_ok = SymEnv::new();
    env_ok.bind(sym, 5).unwrap();
    let v_ok = view(&storage, &layout, Some(&env_ok));
    assert!(v_ok.is_ok(), "live=5 in [1,8] must pass V14: {:?}", v_ok.err());

    // Out-of-range: live = 99 > capacity 8 ⇒ view() fails (V14 caught).
    let mut env_bad = SymEnv::new();
    env_bad.bind(sym, 99).unwrap();
    let v_bad = view(&storage, &layout, Some(&env_bad));
    assert!(v_bad.is_err(), "live=99 > capacity 8 must fail V14");
    let msg = format!("{}", v_bad.err().unwrap());
    assert!(
        msg.contains("validation failed") || msg.contains("out of range"),
        "error should mention V14 failure, got: {msg}"
    );
}

#[test]
fn symbolic_v14_does_not_bake_resolved_value() {
    // Even with a binding env, the sidecar extent keeps the SYMBOL and the
    // capacity window — the resolved value (5) is NOT written back (P4).
    let sym = SymId(7);
    let storage = cpu_f32(64);
    let layout = symbolic_kv_layout(sym);
    let mut env = SymEnv::new();
    env.bind(sym, 5).unwrap();
    let v = view(&storage, &layout, Some(&env)).expect("view");
    let ext = unsafe { &*v.sidecar.as_ref().unwrap().extents.add(1) };
    assert_eq!(ext.sym_id, 7, "symbol must survive");
    assert_eq!(ext.min, 1);
    assert_eq!(ext.capacity, 8, "capacity stays the max, not the live 5");
    assert_ne!(ext.capacity, 5);
}

// ── T9: bundled storage → IS_BUNDLE + views ────────────────────────────────

#[test]
fn bundled_storage_sets_is_bundle_and_views() {
    use fuel_core_types::storage::{compose_bundle, OutputViewSpec};
    use std::sync::Arc;

    // 2-slot bundle: F32[2,3] (y) + F32[1] (extra) — primary dtype F32.
    let specs = vec![
        OutputViewSpec {
            dtype: DType::F32,
            shape: Shape::from_dims(&[2, 3]),
            layout: Layout::contiguous(Shape::from_dims(&[2, 3])),
            name: Some("y"),
        },
        OutputViewSpec {
            dtype: DType::F32,
            shape: Shape::from_dims(&[1]),
            layout: Layout::contiguous(Shape::from_dims(&[1])),
            name: Some("extra"),
        },
    ];
    let (total_bytes, views) = compose_bundle(&specs).expect("compose");
    let storage = Storage::new_bundled(
        BackendStorage::Cpu(CpuStorageBytes::from_zero_bytes(total_bytes)),
        DType::F32,
        Arc::from(views.into_boxed_slice()),
    )
    .expect("bundled storage");
    assert!(storage.is_bundled());

    // The view describes the whole bundle as a 1-D uint8 byte buffer.
    let layout = Layout::contiguous(Shape::from_dims(&[total_bytes]));
    let v = view(&storage, &layout, None).expect("view");

    let sc = v.sidecar.as_ref().expect("bundle sidecar");
    assert_ne!(sc.flags & FDX_FLAG_IS_BUNDLE, 0);
    assert_eq!(sc.views_count, 2);

    // Base is the honest uint8 buffer.
    assert_eq!(v.dl.dtype.code, dtype_code::K_DL_UINT);
    assert_eq!(v.dl.dtype.bits, 8);

    // The two FDXOutputViews mirror the slots.
    let fdx_views = unsafe { core::slice::from_raw_parts(sc.views, sc.views_count as usize) };
    assert_eq!(fdx_views[0].ndim, 2);
    assert_eq!(&fdx_views[0].shape[..2], &[2u64, 3]);
    assert_ne!(fdx_views[0].name_hash, 0); // "y" hashed
    assert_eq!(fdx_views[1].ndim, 1);
    assert_eq!(fdx_views[1].shape[0], 1);

    v.validate().expect("bundle view validates");
}

// ── RankExceeds6 ────────────────────────────────────────────────────────────

#[test]
fn rank_exceeds_6_is_typed_error() {
    let storage = cpu_f32(2 * 2 * 2 * 2 * 2 * 2 * 2);
    let layout = Layout::contiguous(Shape::from_dims(&[2, 2, 2, 2, 2, 2, 2])); // rank 7
    let err = view(&storage, &layout, None).err().expect("rank 7 must error");
    assert!(format!("{err}").contains("exceeds 6"));
}

// ── Quant sidecar from SType (step 3) ───────────────────────────────────────

/// A GGML-block SType projects to an inline-scale FDX GGML_BLOCK sidecar
/// (self-contained, no scale sibling) and validates end-to-end. The base
/// DLTensor is the honest uint8 byte buffer (V3: meaning-bearing ⇒ base uint8).
#[test]
fn ggml_block_stype_emits_inline_quant() {
    use fuel_core_types::{Encoding, GgmlDType, SType};
    let storage = cpu_bytes(DType::F32, 18) // one Q4_0 block (18 bytes)
        .with_stype(SType::from_layer(Encoding::GgmlBlock { ggml_dtype: GgmlDType::Q4_0 }));
    let layout = Layout::contiguous(Shape::from_dims(&[18]));
    let v = view(&storage, &layout, None).expect("view");

    let sc = v.sidecar.as_ref().expect("GGML must emit a sidecar");
    assert_ne!(sc.flags & FDX_FLAG_HAS_QUANT, 0, "HAS_QUANT must be set");
    assert_eq!(sc.quant.family, FDX_QUANT_GGML_BLOCK);
    assert_eq!(sc.quant.scale_present, 0);
    assert_eq!(sc.quant.scale_placement, FDX_SCALE_PLACEMENT_INLINE);
    assert_eq!(sc.quant.scale_buffer, FDX_BUFFER_INLINE);

    // Honesty: the base is the {kDLUInt,8,1} byte stand-in (V3).
    assert_eq!(v.dl.dtype.code, fuel_core_types::dlpack::abi::dtype_code::K_DL_UINT);
    assert_eq!(v.dl.dtype.bits, 8);

    v.validate().expect("GGML_BLOCK sidecar must pass FDX validators");
}

/// An AFFINE_BLOCK (NF4) SType projects to FDX AFFINE_BLOCK with
/// scale_placement=SEPARATE_BUFFER and HAS_QUANT set. A bare `view()` (no op
/// context) emits the SCHEME with `scale_buffer = FDX_BUFFER_NONE` — the scale
/// operand is bound by the consuming op (next step), so the validator flags the
/// unbound scale buffer until then.
#[test]
fn affine_block_stype_emits_scheme_scale_unbound() {
    use fuel_core_types::{Encoding, ScaleSpec, SType};
    use fuel_core_types::ScaleGranularity;
    let storage = cpu_bytes(DType::F4, 64).with_stype(SType::from_layer(
        Encoding::AffineBlock {
            packed: DType::F4,
            block_shape: smallvec::smallvec![64],
            scale: ScaleSpec { dtype: DType::F32, granularity: ScaleGranularity::PerChannel },
            zero_point: None,
        },
    ));
    let layout = Layout::contiguous(Shape::from_dims(&[64]));
    let v = view(&storage, &layout, None).expect("view");

    let sc = v.sidecar.as_ref().expect("AFFINE_BLOCK must emit a sidecar");
    assert_ne!(sc.flags & FDX_FLAG_HAS_QUANT, 0, "HAS_QUANT must be set");
    assert_eq!(sc.quant.family, FDX_QUANT_AFFINE_BLOCK);
    assert_eq!(sc.quant.scale_present, 1);
    assert_eq!(sc.quant.scale_placement, FDX_SCALE_PLACEMENT_SEPARATE_BUFFER);
    assert_eq!(sc.quant.scale_buffer, FDX_BUFFER_NONE, "scale unbound until op binds it");
    assert_eq!(sc.quant.block_ndim, 1);
    assert_eq!(sc.quant.block_shape[0], 64);

    // The SCHEME is described, but the scale operand is not yet bound, so the
    // validator correctly reports the out-of-range scale buffer (the documented
    // step-4 boundary: the consuming op binds the sibling via view_with_quant).
    assert!(v.validate().is_err(), "unbound AFFINE scale must fail validation");
}

/// A plain Storage still emits NO sidecar (byte-identical to pre-SType).
#[test]
fn plain_storage_still_no_quant_sidecar() {
    let storage = cpu_f32(12);
    let layout = Layout::contiguous(Shape::from_dims(&[3, 4]));
    let v = view(&storage, &layout, None).expect("view");
    assert!(v.sidecar.is_none(), "plain storage must stay sidecar-free");
}
