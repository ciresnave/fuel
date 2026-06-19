//! Borrowed, zero-copy DLPack + FDX-sidecar **view** over a Fuel
//! `(Storage, Layout[, SymEnv])` triple at the kernel-call boundary.
//!
//! This is the as-built realization of §2 of
//! `docs/session-prompts/dlpack-comm-layer-plan.md` ("Constructing the DLTensor
//! + FDXSidecar VIEW from `(Storage, Layout)` — no storage rewrite"). The view
//! is a thin, borrowed projection: **nothing about `Storage` or `Layout`
//! changes**, the bytes are never copied, and the `data` pointer is borrowed
//! from the live `Storage` (lifetime-tied via `PhantomData`).
//!
//! # Honesty invariant (FDX §3, P1)
//!
//! The base [`DLTensor`] is *always* honestly interpretable as standard,
//! conformant DLPack on its own:
//! - faithful `dtype` for standard dtypes (F32 → `{kDLFloat,32,1}`, …);
//! - the `{kDLUInt,8,1}` honesty stand-in for sub-byte payloads
//!   (`size_in_bytes()==0`), with the true bit-width/packing only in the
//!   sidecar (§3.4, P5);
//! - explicit, never-NULL strides (§3.2 / V11) — row-major for a contiguous
//!   layout, signed strides (negatives first-class, §3.2.1) passed through
//!   unchanged;
//! - the logical start carried in `byte_offset` (§3.3), never folded into
//!   `data`.
//!
//! # Self-referential-but-sound shape/strides
//!
//! `DLTensor.shape`/`DLTensor.strides` are raw `*mut i64` that must point at the
//! `ndim` shape/stride values for as long as a consumer reads them. The view
//! *owns* those values inline in [`DlpackView::_shape`] / [`DlpackView::_strides`].
//! A naive "store a pointer into my own field" is unsound: moving the
//! `DlpackView` (which Rust may do freely) would dangle the pointer. We avoid
//! that entirely: the stored [`DlpackView::dl`] keeps `shape`/`strides` **NULL**,
//! and the pointers are materialized *on demand* from `&self` by
//! [`DlpackView::dl_tensor`], so they are only ever valid for the duration of a
//! `&self` borrow — exactly the borrow under which a kernel reads them. The
//! validator surface ([`DlpackView::validate`] / [`DlpackView::validate_realize`])
//! uses `dl_tensor()` so it sees the live pointers.
//!
//! # Scope of this slice
//!
//! Sidecar cases derivable from `(storage, layout, env)` alone are built here:
//! sub-byte dtype-ext, symbolic extents, and multi-output bundles. The **quant**
//! (`FDX_FLAG_HAS_QUANT`) and **gather** (`FDX_FLAG_HAS_GATHER`) sidecars need
//! the consuming op's quant params / paged-pool geometry — which `view()` does
//! not receive — and are deliberately deferred (`[consumer-ahead]`). See
//! [`view`] for the documented extension point.

use core::ffi::c_void;
use core::marker::PhantomData;

use fuel_core_types::dlpack::abi::{
    device_type, DLDataType, DLDevice, DLTensor,
};
use fuel_core_types::dlpack::codes::*;
use fuel_core_types::dlpack::convert::{dl_dtype, extent_to_fdx};
use fuel_core_types::dlpack::sidecar::{
    FDXAffine, FDXAffineTerm, FDXBufferRef, FDXDTypeExt, FDXExtent, FDXIndexedResidency,
    FDXOutputView, FDXQuant, FDXResidency, FDXSidecar, FDXStorage, FDXSymBinding, FDXSymEnv,
    FDXTiling,
};
use fuel_core_types::dlpack::validate::{self, FdxValidationError};
use fuel_core_types::shape::Extent;
use fuel_core_types::storage::OutputView;
use fuel_core_types::{DType, DeviceLocation, Error, Layout, Result, SymEnv};

use crate::Storage;

/// Maximum rank an inline DLPack view supports (mirrors Fuel's `DimVec` inline
/// capacity of 6). A layout of higher rank is a typed `RankExceeds6` error.
const MAX_RANK: usize = 6;

/// `FDXPacking` codes (spec §6.1). Mirrored locally because the `dlpack::codes`
/// module does not (yet) export named packing constants.
const PACKING_BYTE_ALIGNED: u8 = 0;
const PACKING_DENSE_SUBBYTE: u8 = 1;

/// A borrowed DLPack + FDX view over a Fuel `(Storage, Layout)` pair.
///
/// Holds the base [`DLTensor`] (with its `shape`/`strides` pointers left NULL —
/// see the module docs and [`DlpackView::dl_tensor`]), an optional owned
/// [`FDXSidecar`], the inline backing arrays the shape/strides point into, and
/// `PhantomData` tying the whole thing to the borrowed `Storage` + `Layout` so
/// the `data` pointer cannot dangle. Constructed per kernel call via [`view`];
/// never persisted; never owns the bytes.
pub struct DlpackView<'a> {
    /// Base DLTensor. `data`/`device`/`ndim`/`dtype`/`byte_offset` are filled;
    /// `shape`/`strides` are **NULL here** and materialized on demand by
    /// [`Self::dl_tensor`] (move-safe — see module docs).
    pub dl: DLTensor,
    /// `None` ⇒ plain, fully-faithful DLPack (P2 absence state). `Some` carries
    /// the non-standard meaning (sub-byte dtype-ext, symbolic extents, bundle).
    pub sidecar: Option<FDXSidecar>,

    /// Backing store for `dl.shape` (capacity bounds; `ndim` entries valid).
    _shape: [i64; MAX_RANK],
    /// Backing store for `dl.strides` (signed; negatives first-class).
    _strides: [i64; MAX_RANK],
    /// Owned FDX extents the sidecar's `extents` pointer references (live form).
    _extents: Vec<FDXExtent>,
    /// Owned FDX buffer table the sidecar's `buffers` pointer references.
    _buffers: Vec<FDXBufferRef>,
    /// Owned FDX bundle views the sidecar's `views` pointer references.
    _views: Vec<FDXOutputView>,

    _marker: PhantomData<(&'a Storage, &'a Layout)>,
}

impl<'a> DlpackView<'a> {
    /// Number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.dl.ndim.max(0) as usize
    }

    /// The capacity shape (`ndim` entries) — what `dl.shape` points into.
    pub fn shape(&self) -> &[i64] {
        &self._shape[..self.ndim()]
    }

    /// The signed strides (`ndim` entries) — what `dl.strides` points into.
    /// Negatives are first-class (FDX §3.2.1).
    pub fn strides(&self) -> &[i64] {
        &self._strides[..self.ndim()]
    }

    /// A complete [`DLTensor`] with `shape`/`strides` pointed at this view's own
    /// inline backing arrays. The returned pointers are valid for the lifetime
    /// of the `&self` borrow — hand it to a consumer that reads it within that
    /// borrow (the kernel-call scope). Move-safe: the view never stores these
    /// pointers, so moving the view cannot dangle them.
    pub fn dl_tensor(&self) -> DLTensor {
        let (shape, strides) = if self.dl.ndim != 0 {
            (
                self._shape.as_ptr() as *mut i64,
                self._strides.as_ptr() as *mut i64,
            )
        } else {
            (core::ptr::null_mut(), core::ptr::null_mut())
        };
        DLTensor {
            data: self.dl.data,
            device: self.dl.device,
            ndim: self.dl.ndim,
            dtype: self.dl.dtype,
            shape,
            strides,
            byte_offset: self.dl.byte_offset,
        }
    }

    /// The live FDX buffer table (`buffers[0]` is the base data buffer).
    pub fn buffers(&self) -> &[FDXBufferRef] {
        &self._buffers
    }

    /// Run every applicable **build-time** FDX validator over this view (the
    /// honesty/coherence checks: V1–V13, gather/extent build arms). No-op for a
    /// `sidecar == None` view (a plain faithful DLPack tensor needs no FDX
    /// validation; V11/V13 still hold structurally by construction).
    pub fn validate(&self) -> std::result::Result<(), FdxValidationError> {
        match &self.sidecar {
            Some(sc) => validate::validate(sc, &self.dl_tensor(), &self._buffers),
            None => Ok(()),
        }
    }

    /// Run the build-time checks plus the **realize-time** symbol-bound checks
    /// (V14 + affine eval) against `env`. Use this when `env` is `Some` and the
    /// view carries symbolic extents — the resolved value is checked
    /// `min <= value <= capacity` but is **never baked into the sidecar** (the
    /// symbol stays, P4).
    pub fn validate_realize(
        &self,
        env: &FDXSymEnv,
    ) -> std::result::Result<(), FdxValidationError> {
        match &self.sidecar {
            Some(sc) => validate::validate_realize(sc, &self.dl_tensor(), &self._buffers, env),
            None => Ok(()),
        }
    }
}

/// FDX device-type code + device id from a backend `DeviceLocation` (§6.6, the
/// coarse DLPack `(device_type, device_id)`; the finer substrate rides the
/// sidecar's `FDXResidency`, built separately).
fn dl_device(loc: DeviceLocation) -> DLDevice {
    let (device_type, device_id) = match loc {
        DeviceLocation::Cpu => (device_type::K_DL_CPU, 0),
        DeviceLocation::Cuda { gpu_id } => (device_type::K_DL_CUDA, gpu_id as i32),
        DeviceLocation::Vulkan { gpu_id } => (device_type::K_DL_VULKAN, gpu_id as i32),
        DeviceLocation::Metal { gpu_id } => (device_type::K_DL_METAL, gpu_id as i32),
    };
    DLDevice { device_type, device_id }
}

/// The base raw data pointer + device location for a [`Storage`], per backend
/// variant. **No offset is folded in** — the intra-buffer start rides
/// `byte_offset` (§3.3). The pointer is metadata only; for a device buffer it is
/// the device-pointer value and must never be dereferenced on the host.
///
/// Only the CPU arm is wired in this slice; the device arms (CUDA/Vulkan/Metal)
/// need each backend's device-pointer accessor and are a later slice
/// (`[consumer-ahead]`). They return a typed error rather than fabricating a
/// wrong pointer (never-panic, no-silent-fallback).
fn base_ptr_and_device(storage: &Storage) -> Result<(*mut c_void, DeviceLocation)> {
    // CPU is always built; the GPU arms are feature-gated. `dispatch_storage!`
    // expands to whichever variants are compiled in. We special-case CPU and
    // route the rest through the deferred error.
    match &storage.inner {
        crate::BackendStorage::Cpu(cpu) => {
            Ok((cpu.bytes().as_ptr() as *mut c_void, DeviceLocation::Cpu))
        }
        // The remaining (feature-gated) arms: device-pointer extraction is a
        // later slice. `dispatch_storage!` documents the full variant set; we
        // produce a typed, non-panicking error here so the CPU path is sound and
        // the GPU path is explicitly TODO rather than silently wrong.
        #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
        other => {
            let _ = crate::dispatch_storage!(other, _inner => _inner.len_bytes());
            Err(Error::Msg(
                "DlpackView: device-pointer extraction for GPU storage is a later \
                 comm-layer slice (CPU is wired); [consumer-ahead]"
                    .into(),
            )
            .bt())
        }
    }
}

/// Whether `d` is a sub-byte dtype that must ride the `{kDLUInt,8,1}` honesty
/// stand-in (FDX §3, §3.4, P5). These are exactly the dtypes whose
/// `size_in_bytes() == 0` (`F4`/`F6E2M3`/`F6E3M2`).
fn is_sub_byte(d: DType) -> bool {
    d.size_in_bytes() == 0
}

/// `(bit_width, packing)` for a dtype's `FDXDTypeExt` (spec §6.1 table). Only
/// called for the stand-in (sub-byte) cases here.
fn sub_byte_bits_and_packing(d: DType) -> (u16, u8) {
    match d {
        DType::F4 => (4, PACKING_DENSE_SUBBYTE),
        DType::F6E2M3 | DType::F6E3M2 => (6, PACKING_DENSE_SUBBYTE),
        // Not reached for the size==0 set, but kept exhaustive-friendly: a byte
        // dtype packs one element per byte.
        _ => (8, PACKING_BYTE_ALIGNED),
    }
}

/// The physical packed byte width of one logical element, for `byte_offset`
/// sizing (FDX §3, P5; the watch-item in §11 of the plan). For a sub-byte dtype
/// the base is `uint8` and the physical width is 1 byte (the packed byte the
/// element lives in); for a standard dtype it is `size_in_bytes()`.
fn physical_elem_bytes(d: DType) -> usize {
    let s = d.size_in_bytes();
    if s == 0 { 1 } else { s }
}

/// All-zero / NONE FDX sub-structs (the "absent" state) used to assemble a
/// sidecar that sets only the flags it means.
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
        scale_buffer: FDX_BUFFER_NONE,
        zp_present: 0,
        zp_dtype: FDX_DTYPE_NONE,
        _pad3: 0,
        zp_buffer: FDX_BUFFER_NONE,
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

fn residency_for(loc: DeviceLocation) -> FDXResidency {
    let (tier, substrate, backend_id, device_index) = match loc {
        DeviceLocation::Cpu => (FDX_TIER_HOST, FDX_SUBSTRATE_HOST_BYTES, FDX_BACKEND_CPU, 0),
        DeviceLocation::Cuda { gpu_id } => (
            FDX_TIER_DEVICE,
            FDX_SUBSTRATE_CUDA_UNTYPED,
            FDX_BACKEND_CUDA,
            gpu_id as u32,
        ),
        DeviceLocation::Vulkan { gpu_id } => (
            FDX_TIER_DEVICE,
            FDX_SUBSTRATE_VULKAN_BUFFER,
            FDX_BACKEND_VULKAN,
            gpu_id as u32,
        ),
        DeviceLocation::Metal { gpu_id } => (
            FDX_TIER_DEVICE,
            FDX_SUBSTRATE_METAL_BUFFER,
            FDX_BACKEND_METAL,
            gpu_id as u32,
        ),
    };
    FDXResidency {
        tier,
        substrate,
        backend_id,
        _pad: 0,
        device_index,
        is_mmap_view: 0,
        _pad2: [0; 7],
        reserved: [0; 4],
    }
}

fn storage_none() -> FDXStorage {
    FDXStorage {
        class: FDX_STORAGE_TRANSIENT,
        _pad: [0; 3],
        _pad_align: 0,
        session_id: FDX_SESSION_NONE as u64,
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
        block_table: fuel_core_types::dlpack::sidecar::FDXBlockTable {
            table_buffer: 0,
            id_dtype: FDX_DTYPE_NONE,
            _pad0: 0,
            max_blocks_per_seq: 0,
            unmapped_sentinel: 0,
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
        logical_extents: [zero_extent(); 6],
        context_lens_buffer: FDX_BUFFER_NONE,
        context_len_sym: FDX_SYM_NONE,
        context_len_scope: 0,
        _pad6: [0; 3],
        reserved: [0; 6],
    }
}

/// An all-zero [`FDXExtent`] (Scalar(0)) for filling `logical_extents` slots.
const fn zero_extent() -> FDXExtent {
    FDXExtent {
        kind: FDX_EXTENT_SCALAR as u8,
        _pad: [0; 3],
        min: 0,
        capacity: 0,
        sym_id: FDX_SYM_NONE,
        sym_scope: 0,
        _pad2: [0; 3],
        cap_kind: FDX_CAP_KIND_EXPLICIT as u8,
        _pad3: [0; 3],
        _pad4: 0,
        affine: FDXAffine {
            c0: 0,
            term_count: 0,
            _pad: [0; 7],
            terms: [FDXAffineTerm { coeff: 0, sym_id: FDX_SYM_NONE, _pad: 0 }; FDX_AFFINE_MAX_TERMS],
        },
        reserved: [0; 2],
    }
}

/// Map a Fuel [`OutputView`] bundle slot to an [`FDXOutputView`] (FDX §6.8 /
/// FKC §5.5). Rank > 6 ⇒ `BundleRankExceeds6` (here surfaced as a typed
/// `Error::Msg`). The slot name is reduced to a stable FNV-1a hash side-table
/// entry (`name_hash`); 0 ⇒ anonymous.
fn output_view_to_fdx(ov: &OutputView) -> Result<FDXOutputView> {
    let dims = ov.shape.dims();
    let ndim = dims.len();
    if ndim > MAX_RANK {
        return Err(Error::Msg(format!(
            "DlpackView bundle slot rank {ndim} exceeds 6 (BundleRankExceeds6)"
        ))
        .bt());
    }
    let mut shape = [0u64; MAX_RANK];
    let mut strides = [0i64; MAX_RANK];
    let st = ov.layout.stride();
    for i in 0..ndim {
        shape[i] = dims[i] as u64;
        strides[i] = st[i] as i64;
    }
    Ok(FDXOutputView {
        byte_offset: ov.byte_offset as u64,
        len_elements: ov.len_elements as u64,
        dtype: dtype_to_fdx_code(ov.dtype),
        _pad: [0; 2],
        ndim: ndim as u32,
        shape,
        strides,
        name_hash: ov.name.map_or(0, fnv1a),
        reserved: [0; 4],
    })
}

/// FNV-1a (64-bit) of a slot name for the bundle name side-table.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// FDX logical-dtype code for a Fuel dtype (the §6.1 table), via the
/// `fuel-core-types` conversion.
fn dtype_to_fdx_code(d: DType) -> u16 {
    fuel_core_types::dlpack::convert::dtype_to_fdx(d)
}

/// Construct a borrowed DLPack + FDX view over `(storage, layout)` with **no
/// copy and no storage rewrite** (plan §2.2). `env` is `Some` for symbolic axes
/// (it runs the realize-time V14 bound check but never bakes the resolved value
/// into the sidecar — the symbol stays, P4).
///
/// Field mapping (§2.2):
/// - `dl.data` = the per-backend raw base pointer (never offset-folded);
/// - `dl.device` = backend variant + `DeviceLocation` device id;
/// - `dl.ndim`/shape = `layout.dims()` (**capacity**; `RankExceeds6` beyond 6);
/// - `dl.strides` = `layout.stride()` cast `isize → i64` (negatives pass through
///   unchanged; explicit, never NULL);
/// - `dl.byte_offset` = `layout.start_offset() * physical_elem_bytes(dtype)`
///   (sub-byte uses the packed physical byte width, §11 watch-item);
/// - `dl.dtype` = faithful [`dl_dtype`] for standard dtypes, or the
///   `{kDLUInt,8,1}` honesty stand-in for sub-byte.
///
/// Sidecar (§2.3) is `Some` iff any of: sub-byte dtype, `layout.has_dynamic()`,
/// or `storage.is_bundled()`; otherwise `None` (a plain faithful tensor).
///
/// # Deferred (`[consumer-ahead]`)
///
/// The **quant** (`FDX_FLAG_HAS_QUANT` / `FDXQuant`) and **gather**
/// (`FDX_FLAG_HAS_GATHER` / `FDXIndexedResidency`) sidecars need the consuming
/// op's quant params / paged-pool geometry, which `view()` does not receive. A
/// richer `view_with_quant(...)` / `view_with_gather(...)` taking that op
/// context is the extension point; do not synthesize those flags here.
pub fn view<'a>(
    storage: &'a Storage,
    layout: &'a Layout,
    env: Option<&SymEnv>,
) -> Result<DlpackView<'a>> {
    let dtype = storage.dtype;
    let dims = layout.dims();
    let ndim = dims.len();
    if ndim > MAX_RANK {
        return Err(Error::Msg(format!(
            "DlpackView: layout rank {ndim} exceeds 6 (RankExceeds6)"
        ))
        .bt());
    }

    let (data, loc) = base_ptr_and_device(storage)?;
    let device = dl_device(loc);

    let sub_byte = is_sub_byte(dtype);
    let bundled = storage.is_bundled();
    let symbolic = layout.has_dynamic();
    let need_sidecar = sub_byte || symbolic || bundled;

    // The base `DLTensor` honesty form (FDX §3, §3.4):
    //   - sub-byte / bundle  ⇒ the honest 1-D `uint8` PHYSICAL BYTE buffer
    //     (`shape = [len_bytes]`, `strides = [1]`, `dtype = {kDLUInt,8,1}`); the
    //     logical element capacity rides the sidecar (dtype_ext / extents / views).
    //   - faithful standard (incl. symbolic) ⇒ the typed CAPACITY-shaped tensor
    //     (`shape = layout.dims()`, signed strides, faithful dtype).
    let physical_byte_base = sub_byte || bundled;

    let mut shape = [0i64; MAX_RANK];
    let mut strides = [0i64; MAX_RANK];
    let base_ndim: usize;
    let dl_dt: DLDataType;
    let byte_offset: u64;

    if physical_byte_base {
        // Honest uint8 physical byte buffer.
        base_ndim = 1;
        shape[0] = storage.len_bytes() as i64;
        strides[0] = 1;
        dl_dt = DLDataType {
            code: fuel_core_types::dlpack::abi::dtype_code::K_DL_UINT,
            bits: 8,
            lanes: 1,
        };
        // start_offset (elements) → physical bytes (§3.3); sub-byte sizes off
        // the packed physical byte width (§11 watch-item).
        byte_offset = (layout.start_offset() as u64)
            .saturating_mul(physical_elem_bytes(dtype) as u64);
    } else {
        // Faithful typed base: capacity shape + signed strides (§2.2).
        base_ndim = ndim;
        let st = layout.stride();
        for i in 0..ndim {
            shape[i] = dims[i] as i64;
            strides[i] = st[i] as i64;
        }
        dl_dt = dl_dtype(dtype);
        byte_offset = (layout.start_offset() as u64)
            .saturating_mul(dtype.size_in_bytes() as u64);
    }

    let dl = DLTensor {
        data,
        device,
        ndim: base_ndim as i32,
        dtype: dl_dt,
        // NULL here — materialized by `dl_tensor()` (move-safe, see module docs).
        shape: core::ptr::null_mut(),
        strides: core::ptr::null_mut(),
        byte_offset,
    };

    if !need_sidecar {
        // Plain, fully-faithful standard tensor — P2 absence state.
        return Ok(DlpackView {
            dl,
            sidecar: None,
            _shape: shape,
            _strides: strides,
            _extents: Vec::new(),
            _buffers: Vec::new(),
            _views: Vec::new(),
            _marker: PhantomData,
        });
    }

    let mut flags: u32 = 0;
    let mut dtype_ext = dtype_ext_none();
    let mut extents: Vec<FDXExtent> = Vec::new();
    let mut views: Vec<FDXOutputView> = Vec::new();

    // (1) sub-byte dtype → HAS_DTYPE_EXT + FDXDTypeExt.
    if sub_byte {
        flags |= FDX_FLAG_HAS_DTYPE_EXT | FDX_FLAG_MEANING_REQUIRES_EXT;
        let (bit_width, packing) = sub_byte_bits_and_packing(dtype);
        dtype_ext = FDXDTypeExt {
            logical_dtype: dtype_to_fdx_code(dtype),
            bit_width,
            packing,
            lanes: 1,
            sub_byte_bit_order: 0, // LSB-first (element 0 in low nibble).
            _pad: 0,
            reserved: [0; 2],
        };
    }

    // (2) symbolic layout → HAS_SYMBOLIC + extents[] (one per axis).
    if symbolic {
        flags |= FDX_FLAG_HAS_SYMBOLIC;
        for e in layout.shape().extents() {
            extents.push(extent_to_fdx(e));
        }
    }

    // (3) bundle → IS_BUNDLE + views[].
    if bundled {
        flags |= FDX_FLAG_IS_BUNDLE;
        if let Some(slots) = storage.bundle() {
            for ov in slots {
                views.push(output_view_to_fdx(ov)?);
            }
        }
    }

    // buffers[0] is ALWAYS the base data buffer (§7.4). It mirrors the base
    // DLTensor's honesty form (already computed above into `shape`/`strides`/
    // `base_ndim`): the honest uint8 physical byte buffer for sub-byte/bundle,
    // else the typed capacity walk.
    let buf0_size = storage.len_bytes() as u64;
    let mut buf0_shape = [0u64; MAX_RANK];
    let mut buf0_strides = [0i64; MAX_RANK];
    for i in 0..base_ndim {
        buf0_shape[i] = shape[i] as u64;
        buf0_strides[i] = strides[i];
    }
    let buffers = vec![FDXBufferRef {
        role: FDX_BUFFER_ROLE_DATA,
        _pad: [0; 1],
        dtype: if physical_byte_base { FDX_DTYPE_U8 } else { dtype_to_fdx_code(dtype) },
        _pad2: 0,
        data,
        device,
        byte_offset,
        size_bytes: buf0_size,
        ndim: base_ndim as u32,
        _pad3: 0,
        shape: buf0_shape,
        strides: buf0_strides,
        reserved: [0; 4],
    }];

    let extents_count = extents.len() as u32;
    let views_count = views.len() as u32;

    // Assemble the sidecar. Pointers reference the OWNED Vecs stored on the view
    // below (set after the Vecs find their final home — see the fixup).
    let sidecar = FDXSidecar {
        magic: FDX_MAGIC,
        version: FDX_VERSION_1,
        struct_bytes: core::mem::size_of::<FDXSidecar>() as u32,
        flags,
        dtype_ext,
        quant: quant_none(),
        extents_count,
        _pad0: 0,
        extents: core::ptr::null(),
        tiling: tiling_none(),
        residency: residency_for(loc),
        storage: storage_none(),
        buffers_count: 1,
        _pad1: 0,
        buffers: core::ptr::null(),
        views_count,
        _pad2: 0,
        views: core::ptr::null(),
        gather: gather_none(),
        reserved: [0; 2],
    };

    let mut v = DlpackView {
        dl,
        sidecar: Some(sidecar),
        _shape: shape,
        _strides: strides,
        _extents: extents,
        _buffers: buffers,
        _views: views,
        _marker: PhantomData,
    };

    // Fix up the sidecar array pointers to reference the view's OWNED Vecs.
    // Like shape/strides, these are only sound while the view is borrowed; the
    // validator surface reads them under `&self`, and they are recomputed on
    // each `validate*` call so a move never leaves a stale pointer in play —
    // but to keep the stored sidecar's pointers honest for direct field access,
    // we point them at the Vec backing here (the Vec is heap-owned, so it does
    // NOT move when the view moves; only the inline shape/strides are
    // move-sensitive, hence those stay NULL until `dl_tensor()`).
    if let Some(sc) = v.sidecar.as_mut() {
        sc.extents = if v._extents.is_empty() { core::ptr::null() } else { v._extents.as_ptr() };
        sc.buffers = if v._buffers.is_empty() { core::ptr::null() } else { v._buffers.as_ptr() };
        sc.views = if v._views.is_empty() { core::ptr::null() } else { v._views.as_ptr() };
    }

    // When `env` is provided AND the view is symbolic, run the realize-time V14
    // bound check (min <= env value <= capacity). We translate the Fuel SymEnv
    // into the boundary FDXSymEnv form for the validator. The resolved value is
    // checked but NEVER written back into the sidecar (P4).
    if symbolic {
        if let Some(env) = env {
            let bindings = collect_bindings(layout, env);
            let fdx_env = FDXSymEnv {
                count: bindings.len() as u32,
                _pad: 0,
                bindings: if bindings.is_empty() {
                    core::ptr::null()
                } else {
                    bindings.as_ptr()
                },
            };
            // `bindings` must outlive the validate call — it does (it lives to
            // the end of this scope, and validate_realize does not retain it).
            v.validate_realize(&fdx_env).map_err(|e| {
                Error::Msg(format!("DlpackView: FDX realize-time validation failed: {e}")).bt()
            })?;
        }
    }

    Ok(v)
}

/// Collect the `SymId → value` bindings for every symbolic axis of `layout`'s
/// shape into the boundary `FDXSymBinding` form (sorted by sym id). Unbound
/// symbols are simply omitted; the validator then reports `UnboundSymbol`.
fn collect_bindings(layout: &Layout, env: &SymEnv) -> Vec<FDXSymBinding> {
    let mut out: Vec<FDXSymBinding> = Vec::new();
    for e in layout.shape().extents() {
        if let Extent::Range { sym, .. } = e {
            if let Some(v) = env.get(sym) {
                if !out.iter().any(|b| b.sym_id == sym.0) {
                    out.push(FDXSymBinding { sym_id: sym.0, _pad: 0, value: v as u64 });
                }
            }
        }
    }
    out.sort_unstable_by_key(|b| b.sym_id);
    out
}

#[cfg(test)]
mod tests;
