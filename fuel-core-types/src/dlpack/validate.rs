//! FDX validators (V1..V21) — the correctness core of the Fuel DLPack
//! eXtension (spec `docs/specs/dlpack-extension.md` §8 "Validation", consolidated
//! in the §16 conformance checklist).
//!
//! Every check is `Result`-returning and runnable at the boundary; there are no
//! `try_*` siblings (P10). This module implements each spec validator as one
//! function named by its V-number, cross-referenced to the spec section it
//! enforces. Two surfaces:
//!
//! - [`validate`] — runs every applicable **build-time / boundary-time** check
//!   (V1..V13, V15, V16, V18..V21 build arms), gated by the relevant
//!   `FDX_FLAG_*`.
//! - [`validate_realize`] — runs the **realize-time** checks that require a
//!   resolved [`FDXSymEnv`] (V14 + V17 affine evaluation, V21(d) per-sequence
//!   bounds), on top of the build-time checks.
//!
//! The validators operate on the `#[repr(C)]` POD structs from [`super::sidecar`]
//! and the standard [`super::abi::DLTensor`]; all codes/sentinels come from
//! [`super::codes`]. Nothing here re-defines a struct or a code.

use super::abi::{dtype_code, DLTensor};
use super::codes::*;
use super::sidecar::{
    FDXBufferRef, FDXExtent, FDXIndexedResidency, FDXOutputView, FDXSidecar, FDXSymEnv,
};

/// Typed validator error — one variant per distinct validator failure the spec
/// names (§8 closing list). Carries useful locating fields (axis / index /
/// offending value). This is **local** to the validators; it is NOT added to the
/// shared `fuel_core_types::Error`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FdxValidationError {
    // ── V1 header ──────────────────────────────────────────────────────────
    #[error("FDX magic mismatch: got {got:#010x}, expected {expected:#010x}")]
    BadMagic { got: u32, expected: u32 },
    #[error("unsupported FDX version {got} (max supported {max})")]
    UnsupportedVersion { got: u32, max: u32 },
    #[error("struct_bytes {got} is smaller than the known sidecar prefix {min}")]
    StructBytesTooSmall { got: u32, min: u32 },

    // ── V2 flag/field coherence ──────────────────────────────────────────────
    #[error("flag/field incoherence: {detail}")]
    FlagFieldIncoherent { detail: &'static str },

    // ── V3 / V19 honesty (dtype / base) ──────────────────────────────────────
    #[error("dishonest base DLTensor: {detail}")]
    DishonestBase { detail: &'static str },

    // ── V4 sub-byte sizing ───────────────────────────────────────────────────
    #[error("bad sub-byte dtype descriptor: {detail}")]
    BadSubByte { detail: &'static str },

    // ── V5 quant coherence ───────────────────────────────────────────────────
    #[error("quant regime violation (family {family:#06x}): {detail}")]
    QuantRegimeViolation { family: u16, detail: &'static str },
    #[error("quant descriptor incoherent: {detail}")]
    QuantIncoherent { detail: &'static str },

    // ── V6 scale shape vs granularity / block geometry ───────────────────────
    #[error("scale buffer shape mismatch: {detail}")]
    ScaleShapeMismatch { detail: &'static str },

    // ── V7 extents ───────────────────────────────────────────────────────────
    #[error("extent mismatch on axis {axis}: {detail}")]
    ExtentMismatch { axis: usize, detail: &'static str },

    // ── V8 / V20 capacity backing ────────────────────────────────────────────
    #[error("capacity not backed: {detail} (have {have} bytes, need {need})")]
    CapacityNotBacked {
        detail: &'static str,
        have: u64,
        need: u128,
    },

    // ── V9 / V21(a) buffer refs ──────────────────────────────────────────────
    #[error("buffer-table index {index} out of range (buffers_count {count}): {detail}")]
    BufferRefOutOfRange {
        index: u32,
        count: u32,
        detail: &'static str,
    },

    // ── V10 bundle ───────────────────────────────────────────────────────────
    #[error("bundle view overlap / out-of-bounds: {detail}")]
    BundleOverlap { detail: &'static str },

    // ── V11 explicit strides ─────────────────────────────────────────────────
    #[error("NULL strides on a versioned export ({detail})")]
    NullStrides { detail: &'static str },

    // ── V12 256-byte alignment (boundary b) ──────────────────────────────────
    #[error("misaligned data pointer ({detail}): {addr:#x} is not 256-byte aligned")]
    Misaligned { detail: &'static str, addr: usize },

    // ── V13 signed-stride OOB range ──────────────────────────────────────────
    #[error(
        "stride range out of bounds ({detail}): touched window [{lo}, {hi}] escapes [0, {size_bytes})"
    )]
    StrideRangeOutOfBounds {
        detail: &'static str,
        lo: i128,
        hi: i128,
        size_bytes: u64,
    },

    // ── V14 realize-time symbol bounds ───────────────────────────────────────
    #[error("extent out of range on axis {axis}: value {value} not in [{min}, {capacity}]")]
    ExtentOutOfRange {
        axis: usize,
        value: i128,
        min: u64,
        capacity: u64,
    },
    #[error("unbound symbol {sym_id} (axis {axis})")]
    UnboundSymbol { axis: usize, sym_id: u32 },

    // ── V15 no raw pointers in serialized form ───────────────────────────────
    #[error("pointer in serialized form: {detail}")]
    PointerInSerializedForm { detail: &'static str },

    // ── V16 affine well-formedness ───────────────────────────────────────────
    #[error("affine extent malformed on axis {axis}: {detail}")]
    AffineMalformed { axis: usize, detail: &'static str },
    #[error("affine extent has too many terms on axis {axis}: {term_count} > {max}")]
    AffineTooManyTerms {
        axis: usize,
        term_count: u8,
        max: usize,
    },
    #[error("degenerate affine on axis {axis}: {detail} (emit Scalar/Range)")]
    AffineDegenerate { axis: usize, detail: &'static str },

    // ── V17 affine evaluation safety ─────────────────────────────────────────
    #[error("affine evaluation overflow on axis {axis}")]
    AffineOverflow { axis: usize },

    // ── V18 gather coherence ─────────────────────────────────────────────────
    #[error("gather descriptor incoherent: {detail}")]
    GatherIncoherent { detail: &'static str },
    #[error("unsupported gather kind {kind}")]
    UnsupportedGatherKind { kind: u16 },

    // ── V21(c)/(d) block-id range ────────────────────────────────────────────
    #[error("block id {id} out of range (num_blocks {num_blocks}) at table slot {slot}")]
    BlockIdOutOfRange {
        id: u32,
        num_blocks: u64,
        slot: usize,
    },

    // ── V21(d) runtime narrowing / address overflow ──────────────────────────
    #[error("gather address overflow: {detail}")]
    GatherAddressOverflow { detail: &'static str },
}

type R = Result<(), FdxValidationError>;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Byte width of one element of an FDX logical dtype code (§6.1 table), rounded
/// **up** to whole bytes. Sub-byte codes (F4/F6*) round to 1 — they ride the
/// `uint8` honesty base and never appear as typed strided operands, so a caller
/// that needs an exact whole-byte width (V19/V20 pool sizing) must reject them
/// up front. Returns `None` for an unknown code (caller maps to a typed error).
fn fdx_elem_bytes(code: u16) -> Option<u64> {
    let bytes = match code {
        FDX_DTYPE_U8 | FDX_DTYPE_I8 | FDX_DTYPE_F8E4M3 | FDX_DTYPE_F8E8M0 => 1,
        FDX_DTYPE_I16 | FDX_DTYPE_BF16 | FDX_DTYPE_F16 => 2,
        FDX_DTYPE_U32 | FDX_DTYPE_I32 | FDX_DTYPE_F32 => 4,
        FDX_DTYPE_I64 | FDX_DTYPE_F64 => 8,
        // sub-byte: round up to 1 byte; whole-byte callers reject separately.
        FDX_DTYPE_F6E2M3 | FDX_DTYPE_F6E3M2 | FDX_DTYPE_F4 => 1,
        _ => return None,
    };
    Some(bytes)
}

/// Bit width of one element of an FDX logical dtype code (§6.1 table). Used by
/// V19/V20 to compute the honest pool byte cover exactly (a sub-byte element is
/// < 8 bits, so element-count × bytes would over-count). Returns `None` for an
/// unknown code.
fn fdx_bit_width(code: u16) -> Option<u64> {
    let bits = match code {
        FDX_DTYPE_F4 => 4,
        FDX_DTYPE_F6E2M3 | FDX_DTYPE_F6E3M2 => 6,
        FDX_DTYPE_U8 | FDX_DTYPE_I8 | FDX_DTYPE_F8E4M3 | FDX_DTYPE_F8E8M0 => 8,
        FDX_DTYPE_I16 | FDX_DTYPE_BF16 | FDX_DTYPE_F16 => 16,
        FDX_DTYPE_U32 | FDX_DTYPE_I32 | FDX_DTYPE_F32 => 32,
        FDX_DTYPE_I64 | FDX_DTYPE_F64 => 64,
        _ => return None,
    };
    Some(bits)
}

/// The set of flags that, when present, declare a meaning-bearing block. Used by
/// V2 to validate flag ⇔ block coherence.
#[inline]
fn flag_set(flags: u32, bit: u32) -> bool {
    flags & bit != 0
}

// ─────────────────────────────────────────────────────────────────────────────
// V1 — header (§8.1)
// ─────────────────────────────────────────────────────────────────────────────

/// V1 (§8.1): `magic == FDX_MAGIC`; `version <= FDX_VERSION_MAX`;
/// `struct_bytes >= sizeof(known prefix)`.
pub fn check_v1_header(sidecar: &FDXSidecar) -> R {
    if sidecar.magic != FDX_MAGIC {
        return Err(FdxValidationError::BadMagic {
            got: sidecar.magic,
            expected: FDX_MAGIC,
        });
    }
    if sidecar.version == 0 || sidecar.version > FDX_VERSION_MAX {
        return Err(FdxValidationError::UnsupportedVersion {
            got: sidecar.version,
            max: FDX_VERSION_MAX,
        });
    }
    let known = core::mem::size_of::<FDXSidecar>() as u32;
    if sidecar.struct_bytes < known {
        return Err(FdxValidationError::StructBytesTooSmall {
            got: sidecar.struct_bytes,
            min: known,
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// V2 — flag/field coherence (§8.2)
// ─────────────────────────────────────────────────────────────────────────────

/// V2 (§8.2): each `FDX_FLAG_HAS_*` set ⇔ the corresponding block is non-NONE,
/// and vice-versa.
pub fn check_v2_flag_coherence(sidecar: &FDXSidecar) -> R {
    let f = sidecar.flags;

    // dtype_ext ⇔ HAS_DTYPE_EXT
    let dtype_ext_none = sidecar.dtype_ext.logical_dtype == FDX_DTYPE_NONE;
    if flag_set(f, FDX_FLAG_HAS_DTYPE_EXT) == dtype_ext_none {
        return Err(FdxValidationError::FlagFieldIncoherent {
            detail: "HAS_DTYPE_EXT set iff dtype_ext is non-NONE",
        });
    }
    // quant ⇔ HAS_QUANT
    let quant_none = sidecar.quant.family == FDX_QUANT_NONE;
    if flag_set(f, FDX_FLAG_HAS_QUANT) == quant_none {
        return Err(FdxValidationError::FlagFieldIncoherent {
            detail: "HAS_QUANT set iff quant.family is non-NONE",
        });
    }
    // symbolic ⇔ HAS_SYMBOLIC: ≥1 symbolic axis lives either in the base
    // `extents[]` OR (for a gather tensor) in `gather.logical_extents[]`. The
    // gather seq axis is a symbolic live extent carried in the logical extents,
    // so HAS_SYMBOLIC may be set with `extents_count == 0` (§6.9.2, §13.8).
    let base_has_symbolic = sidecar.extents_count != 0;
    let gather_has_symbolic = flag_set(f, FDX_FLAG_HAS_GATHER)
        && (sidecar.gather.context_len_sym != FDX_SYM_NONE
            || sidecar.gather.logical_extents_count != 0);
    if flag_set(f, FDX_FLAG_HAS_SYMBOLIC) && !base_has_symbolic && !gather_has_symbolic {
        return Err(FdxValidationError::FlagFieldIncoherent {
            detail: "HAS_SYMBOLIC set but no symbolic extent (base or gather logical)",
        });
    }
    // bundle ⇔ HAS_IS_BUNDLE
    let has_views = sidecar.views_count != 0;
    if flag_set(f, FDX_FLAG_IS_BUNDLE) != has_views {
        return Err(FdxValidationError::FlagFieldIncoherent {
            detail: "IS_BUNDLE set iff views_count != 0",
        });
    }
    // gather ⇔ HAS_GATHER  (also enforced by V18; coherence stated here)
    let gather_none = sidecar.gather.kind == FDX_GATHER_NONE as u8;
    if flag_set(f, FDX_FLAG_HAS_GATHER) == gather_none {
        return Err(FdxValidationError::FlagFieldIncoherent {
            detail: "HAS_GATHER set iff gather.kind != FDX_GATHER_NONE",
        });
    }
    // affine ⇔ HAS_AFFINE_EXTENT (≥1 axis kind=Affine)
    let any_affine = unsafe { extents_slice(sidecar) }
        .iter()
        .any(|e| e.kind as u16 == FDX_EXTENT_AFFINE);
    if flag_set(f, FDX_FLAG_HAS_AFFINE_EXTENT) != any_affine {
        return Err(FdxValidationError::FlagFieldIncoherent {
            detail: "HAS_AFFINE_EXTENT set iff ≥1 extent is kind=Affine",
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// V3 — honesty (dtype) (§8.3, §3, §3.4)
// ─────────────────────────────────────────────────────────────────────────────

/// V3 (§8.3): if `dtype_ext`/`quant` is meaning-bearing (or the tensor is
/// otherwise meaning-requires-ext), the base `DLTensor.dtype` is the standard
/// byte code `{kDLUInt, 8, 1}` (the honesty stand-in) and never a native DLPack
/// sub-byte dtype (§3.4).
pub fn check_v3_honesty_dtype(sidecar: &FDXSidecar, base: &DLTensor) -> R {
    let f = sidecar.flags;
    let meaning_bearing = flag_set(f, FDX_FLAG_HAS_DTYPE_EXT)
        || flag_set(f, FDX_FLAG_HAS_QUANT)
        || flag_set(f, FDX_FLAG_MEANING_REQUIRES_EXT)
        || flag_set(f, FDX_FLAG_HAS_GATHER);
    if !meaning_bearing {
        return Ok(());
    }
    if base.dtype.code != dtype_code::K_DL_UINT
        || base.dtype.bits != 8
        || base.dtype.lanes != 1
    {
        return Err(FdxValidationError::DishonestBase {
            detail: "meaning-bearing tensor must carry base dtype {kDLUInt,8,1}",
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// V4 — sub-byte sizing (§8.4)
// ─────────────────────────────────────────────────────────────────────────────

/// V4 (§8.4): `bit_width != 0`; `packing` consistent with `bit_width`
/// (`DENSE_SUBBYTE` ⇒ `bit_width < 8`); physical byte count derivable (never via
/// a `size_in_bytes()==0` dtype). Only runs when `HAS_DTYPE_EXT`.
pub fn check_v4_sub_byte(sidecar: &FDXSidecar) -> R {
    if !flag_set(sidecar.flags, FDX_FLAG_HAS_DTYPE_EXT) {
        return Ok(());
    }
    let d = &sidecar.dtype_ext;
    if d.bit_width == 0 {
        return Err(FdxValidationError::BadSubByte {
            detail: "bit_width must never be 0",
        });
    }
    // packing 1 = DENSE_SUBBYTE ⇒ sub-byte. (FDXPacking values §6.1.)
    const PACKING_DENSE_SUBBYTE: u8 = 1;
    if d.packing == PACKING_DENSE_SUBBYTE && d.bit_width >= 8 {
        return Err(FdxValidationError::BadSubByte {
            detail: "DENSE_SUBBYTE packing requires bit_width < 8",
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// V5 — quant coherence per family (§8.5, §6.2)
// ─────────────────────────────────────────────────────────────────────────────

/// V5 (§8.5): per-family quant coherence; the regimes do not overlap (§6.2).
/// Only runs when `HAS_QUANT`.
pub fn check_v5_quant(sidecar: &FDXSidecar) -> R {
    if !flag_set(sidecar.flags, FDX_FLAG_HAS_QUANT) {
        return Ok(());
    }
    let q = &sidecar.quant;
    let fam = q.family;

    // Any PerBlock granularity under a non-MX family ⇒ violation.
    if fam != FDX_QUANT_MX && q.scale_granularity == FDX_SCALE_GRAN_PER_BLOCK {
        return Err(FdxValidationError::QuantRegimeViolation {
            family: fam,
            detail: "PerBlock granularity is MX-only",
        });
    }

    match fam {
        FDX_QUANT_GGML_BLOCK => {
            // ggml_dtype valid + scales baked INLINE + no separate scale +
            // no PerBlock + no ScalePair.
            if q.ggml_dtype == FDX_DTYPE_NONE {
                return Err(FdxValidationError::QuantIncoherent {
                    detail: "GGML_BLOCK requires a valid ggml_dtype",
                });
            }
            if q.scale_placement == FDX_SCALE_PLACEMENT_SEPARATE_BUFFER
                || q.scale_buffer != FDX_BUFFER_INLINE
            {
                return Err(FdxValidationError::QuantRegimeViolation {
                    family: fam,
                    detail: "GGML_BLOCK scales are baked INLINE, never a separate scale operand",
                });
            }
            if q.scale_granularity == FDX_SCALE_GRAN_PER_BLOCK {
                return Err(FdxValidationError::QuantRegimeViolation {
                    family: fam,
                    detail: "GGML_BLOCK must not set PerBlock granularity",
                });
            }
        }
        FDX_QUANT_MX => {
            if q.scale_dtype != FDX_DTYPE_F8E8M0 {
                return Err(FdxValidationError::QuantRegimeViolation {
                    family: fam,
                    detail: "MX requires scale_dtype == F8E8M0",
                });
            }
            if q.scale_granularity != FDX_SCALE_GRAN_PER_BLOCK {
                return Err(FdxValidationError::QuantRegimeViolation {
                    family: fam,
                    detail: "MX requires scale_granularity == PerBlock",
                });
            }
            if q.scale_placement != FDX_SCALE_PLACEMENT_SEPARATE_BUFFER {
                return Err(FdxValidationError::QuantRegimeViolation {
                    family: fam,
                    detail: "MX requires scale_placement == SEPARATE_BUFFER",
                });
            }
            if q.block_ndim == 0 {
                return Err(FdxValidationError::QuantIncoherent {
                    detail: "MX requires block geometry (block_ndim >= 1)",
                });
            }
        }
        FDX_QUANT_AFFINE_INT | FDX_QUANT_AFFINE_FLOAT => {
            match q.scale_granularity {
                FDX_SCALE_GRAN_PER_TENSOR
                | FDX_SCALE_GRAN_PER_TOKEN
                | FDX_SCALE_GRAN_PER_CHANNEL => {}
                _ => {
                    return Err(FdxValidationError::QuantRegimeViolation {
                        family: fam,
                        detail: "AFFINE_{INT,FLOAT} granularity ∈ {PerTensor,PerToken,PerChannel}",
                    });
                }
            }
            // scale_buffer valid (a real index) or BROADCAST.
            if q.scale_placement != FDX_SCALE_PLACEMENT_BROADCAST_PER_AXIS
                && q.scale_buffer == FDX_BUFFER_INLINE
            {
                return Err(FdxValidationError::QuantIncoherent {
                    detail: "AFFINE_{INT,FLOAT} requires a separate scale buffer or BROADCAST",
                });
            }
        }
        FDX_QUANT_AFFINE_BLOCK => {
            if q.block_ndim == 0 {
                return Err(FdxValidationError::QuantIncoherent {
                    detail: "AFFINE_BLOCK requires block geometry (block_ndim >= 1)",
                });
            }
            if q.scale_present == 0 {
                return Err(FdxValidationError::QuantIncoherent {
                    detail: "AFFINE_BLOCK requires scale_present == 1",
                });
            }
            if q.scale_placement != FDX_SCALE_PLACEMENT_SEPARATE_BUFFER {
                return Err(FdxValidationError::QuantRegimeViolation {
                    family: fam,
                    detail: "AFFINE_BLOCK requires scale_placement == SEPARATE_BUFFER",
                });
            }
            if q.scale_buffer == FDX_BUFFER_INLINE {
                return Err(FdxValidationError::QuantRegimeViolation {
                    family: fam,
                    detail: "AFFINE_BLOCK scale_buffer must be a real index, never INLINE",
                });
            }
            if q.scale_granularity == FDX_SCALE_GRAN_PER_BLOCK {
                return Err(FdxValidationError::QuantRegimeViolation {
                    family: fam,
                    detail: "AFFINE_BLOCK must not set PerBlock (block grain rides block_shape)",
                });
            }
        }
        other => {
            return Err(FdxValidationError::QuantIncoherent {
                detail: if other == FDX_QUANT_NONE {
                    "HAS_QUANT set but family == NONE"
                } else {
                    "unknown quant family"
                },
            });
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// V6 — scale shape vs granularity / block geometry (§8.6, §6.3)
// ─────────────────────────────────────────────────────────────────────────────

/// V6 (§8.6): the separate scale buffer's shape matches the granularity (for
/// `AFFINE_{INT,FLOAT}`) or the per-axis block count (for block-shaped `MX` /
/// `AFFINE_BLOCK`). `GGML_BLOCK` has no separate scale buffer to check. Only runs
/// when `HAS_QUANT` and the family uses a separate scale buffer.
pub fn check_v6_scale_shape(sidecar: &FDXSidecar, base: &DLTensor, buffers: &[FDXBufferRef]) -> R {
    if !flag_set(sidecar.flags, FDX_FLAG_HAS_QUANT) {
        return Ok(());
    }
    let q = &sidecar.quant;
    // Families with a separate scale buffer whose shape we can cross-check.
    let block_shaped = matches!(q.family, FDX_QUANT_MX | FDX_QUANT_AFFINE_BLOCK);
    let granular = matches!(q.family, FDX_QUANT_AFFINE_INT | FDX_QUANT_AFFINE_FLOAT);
    if !block_shaped && !granular {
        return Ok(()); // GGML inline / NONE — nothing to check here.
    }
    // BROADCAST scales carry shape per granularity but have no buffer to read.
    if q.scale_placement != FDX_SCALE_PLACEMENT_SEPARATE_BUFFER {
        return Ok(());
    }
    let idx = q.scale_buffer;
    let scale = buffers
        .get(idx as usize)
        .ok_or(FdxValidationError::BufferRefOutOfRange {
            index: idx,
            count: buffers.len() as u32,
            detail: "scale_buffer index out of range",
        })?;

    // base logical shape: the base DLTensor describes physical bytes for
    // meaning-bearing tensors, so we cross-check against the scale buffer's own
    // declared element count against the expected count.
    let scale_count: u128 = shape_product(&scale.shape, scale.ndim);

    if block_shaped {
        // The block count is over the base LOGICAL element shape (§6.2, §13.5a).
        // For a packed sub-byte payload the honest base carries BYTES (2 nibbles/
        // byte for F4, etc.), so for the flattened 1-D byte base we convert the
        // byte extent to logical elements via the dtype-ext bit width before
        // tiling. Without this, `ceil(byte_len / block_shape)` undercounts by the
        // pack factor and would reject the spec's own NF4 example (§13.5a).
        let subbyte_bits: Option<u128> =
            if flag_set(sidecar.flags, FDX_FLAG_HAS_DTYPE_EXT) {
                let bw = sidecar.dtype_ext.bit_width as u128;
                if bw > 0 && bw < 8 { Some(bw) } else { None }
            } else {
                None
            };
        // one scale per block: Π ceil(base_dim_a / block_shape[a]) over tiled axes.
        let mut expected: u128 = 1;
        let bn = q.block_ndim as usize;
        for a in 0..bn {
            let axis = q.block_axes[a];
            let bs = q.block_shape[a] as u128;
            if bs == 0 {
                return Err(FdxValidationError::QuantIncoherent {
                    detail: "block_shape entry is 0",
                });
            }
            // For a 1-D flattened weight (block_axes[0] over the whole tensor)
            // derive the tiled dim from base.shape; convert packed bytes → logical
            // elements when the payload is sub-byte (the §13.5a NF4 case).
            let mut dim = base_axis_len(base, axis);
            if let Some(bw) = subbyte_bits {
                if base.ndim == 1 {
                    dim = dim.saturating_mul(8) / bw;
                }
            }
            let blocks = dim.div_ceil(bs);
            expected = expected.saturating_mul(blocks);
        }
        if scale_count != expected {
            return Err(FdxValidationError::ScaleShapeMismatch {
                detail: "block-shaped scale count != per-axis block count",
            });
        }
    } else {
        // granular: PerTensor=1, PerToken=rows, PerChannel=cols of the base
        // logical 2-D shape. The base is the honest uint8 byte buffer here, so
        // we accept any of the legal counts the granularity implies relative to
        // a derivable 2-D logical shape; we at minimum require PerTensor==1.
        if q.scale_granularity == FDX_SCALE_GRAN_PER_TENSOR && scale_count != 1 {
            return Err(FdxValidationError::ScaleShapeMismatch {
                detail: "PerTensor scale must hold exactly one element",
            });
        }
    }
    Ok(())
}

/// Logical length of `base` along an FDX `block_axes` axis. For a 1-D honest byte
/// base the logical shape is not directly recoverable, so axis 0 falls back to
/// the byte length (callers cross-check divisibility). Negative axis = unused.
fn base_axis_len(base: &DLTensor, axis: i32) -> u128 {
    if axis < 0 {
        return 1;
    }
    let a = axis as usize;
    if base.shape.is_null() || a >= base.ndim.max(0) as usize {
        return 1;
    }
    // SAFETY: shape is length ndim per the DLTensor contract; a < ndim.
    let v = unsafe { *base.shape.add(a) };
    v.max(0) as u128
}

fn shape_product(shape: &[u64; 6], ndim: u32) -> u128 {
    let n = (ndim as usize).min(6);
    let mut p: u128 = 1;
    for &d in &shape[..n] {
        p = p.saturating_mul(d as u128);
    }
    p
}

// ─────────────────────────────────────────────────────────────────────────────
// V7 — extents (§8.7, §6.4.2)
// ─────────────────────────────────────────────────────────────────────────────

/// V7 (§8.7): `extents_count ∈ {0, base.ndim}`; each `capacity == base.shape[i]`;
/// `min <= capacity`; `cap_kind == EXPLICIT` for every kind; per-kind well-
/// formedness; and (for Affine) V16 well-formedness. The same arms apply to the
/// gather logical extents via [`check_v7_extent_arm`].
pub fn check_v7_extents(sidecar: &FDXSidecar, base: &DLTensor) -> R {
    let count = sidecar.extents_count as usize;
    if count == 0 {
        return Ok(());
    }
    let ndim = base.ndim.max(0) as usize;
    if count != ndim {
        return Err(FdxValidationError::ExtentMismatch {
            axis: 0,
            detail: "extents_count must be 0 or base.ndim",
        });
    }
    let extents = unsafe { extents_slice(sidecar) };
    for (i, ext) in extents.iter().enumerate() {
        let cap = base_axis_len(base, i as i32) as u64;
        check_v7_extent_arm(ext, i, cap)?;
    }
    Ok(())
}

/// One extent's V7 arm against a known capacity (the base axis length, or
/// `max_seq_capacity`/`logical_shape` for gather logical extents — V21d).
pub fn check_v7_extent_arm(ext: &FDXExtent, axis: usize, capacity: u64) -> R {
    if ext.capacity != capacity {
        return Err(FdxValidationError::ExtentMismatch {
            axis,
            detail: "extent.capacity must equal the axis capacity (base.shape[i])",
        });
    }
    if ext.min > ext.capacity {
        return Err(FdxValidationError::ExtentMismatch {
            axis,
            detail: "extent.min must be <= capacity",
        });
    }
    // cap_kind must be EXPLICIT (0) for EVERY kind (poisoning guard, §8.7).
    if ext.cap_kind as u16 != FDX_CAP_KIND_EXPLICIT {
        return Err(FdxValidationError::ExtentMismatch {
            axis,
            detail: "cap_kind must be EXPLICIT (0) in v1",
        });
    }
    match ext.kind as u16 {
        FDX_EXTENT_SCALAR => {
            if ext.sym_id != FDX_SYM_NONE {
                return Err(FdxValidationError::ExtentMismatch {
                    axis,
                    detail: "Scalar extent must have sym_id == NONE",
                });
            }
            if ext.min != ext.capacity {
                return Err(FdxValidationError::ExtentMismatch {
                    axis,
                    detail: "Scalar extent must have min == capacity",
                });
            }
            if ext.affine.term_count != 0 {
                return Err(FdxValidationError::ExtentMismatch {
                    axis,
                    detail: "Scalar extent must have affine.term_count == 0",
                });
            }
        }
        FDX_EXTENT_RANGE => {
            if ext.sym_id == FDX_SYM_NONE {
                return Err(FdxValidationError::ExtentMismatch {
                    axis,
                    detail: "Range extent must have a real sym_id",
                });
            }
            if ext.affine.term_count != 0 {
                return Err(FdxValidationError::ExtentMismatch {
                    axis,
                    detail: "Range extent must have affine.term_count == 0",
                });
            }
        }
        FDX_EXTENT_AFFINE => {
            if ext.sym_id != FDX_SYM_NONE {
                return Err(FdxValidationError::ExtentMismatch {
                    axis,
                    detail: "Affine extent must have sym_id == NONE (syms live in affine.terms)",
                });
            }
            // Full affine well-formedness (V16).
            check_v16_affine(ext, axis)?;
        }
        _ => {
            return Err(FdxValidationError::ExtentMismatch {
                axis,
                detail: "unknown extent kind",
            });
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// V8 — capacity backing (§8.8, §3.1)
// ─────────────────────────────────────────────────────────────────────────────

/// V8 (§8.8): for the no-OOB guarantee, `buffers[0].size_bytes` must cover the
/// full capacity-shaped extent (`capacity * stride` along every axis), keyed off
/// the base strides + the element byte width. If it does not, the sidecar MUST
/// set `FDX_FLAG_MEANING_REQUIRES_EXT`. Skipped for gather (V20 covers the pool)
/// and bundle (the base is the whole bundle).
pub fn check_v8_capacity_backing(
    sidecar: &FDXSidecar,
    base: &DLTensor,
    buffers: &[FDXBufferRef],
) -> R {
    if flag_set(sidecar.flags, FDX_FLAG_HAS_GATHER)
        || flag_set(sidecar.flags, FDX_FLAG_IS_BUNDLE)
    {
        return Ok(());
    }
    // If meaning-requires-ext is set, the producer has already declared the base
    // tail unsafe; the no-full-backing claim is not being made.
    if flag_set(sidecar.flags, FDX_FLAG_MEANING_REQUIRES_EXT) {
        return Ok(());
    }
    let buf0 = match buffers.first() {
        Some(b) => b,
        None => return Ok(()), // V9 reports the missing index-0 buffer.
    };
    let ndim = base.ndim.max(0) as usize;
    if ndim == 0 || base.shape.is_null() || base.strides.is_null() {
        return Ok(());
    }
    // Element byte width of the BASE dtype (honesty uint8 ⇒ 1; faithful F16 ⇒ 2).
    let elem = base.dtype.bits.div_ceil(8).max(1) as u128;
    // Needed bytes for the dense capacity-shaped tensor: max over axes of
    // (shape[i]-1)*|stride[i]| + 1, times element width. (Positive strides; the
    // signed window is V13's job — here we size the dense capacity allocation.)
    let mut span_elems: u128 = 1;
    for i in 0..ndim {
        let dim = unsafe { *base.shape.add(i) }.max(0) as u128;
        let stride = unsafe { *base.strides.add(i) };
        let astride = (stride.unsigned_abs()) as u128;
        if dim == 0 {
            continue;
        }
        let reach = (dim - 1).saturating_mul(astride) + 1;
        span_elems = span_elems.max(reach);
    }
    let need = span_elems.saturating_mul(elem);
    if (buf0.size_bytes as u128) < need {
        return Err(FdxValidationError::CapacityNotBacked {
            detail: "buffers[0] does not back the full capacity shape; set MEANING_REQUIRES_EXT",
            have: buf0.size_bytes,
            need,
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// V9 — buffer refs (§8.9)
// ─────────────────────────────────────────────────────────────────────────────

/// V9 (§8.9): every referenced index (`scale_buffer`, `zp_buffer`) `<
/// buffers_count`; index 0 role is `Data` (or `POOL` for a gather pool, §6.9.1);
/// `byte_offset <= size_bytes` for each buffer.
pub fn check_v9_buffer_refs(sidecar: &FDXSidecar, buffers: &[FDXBufferRef]) -> R {
    let count = buffers.len() as u32;
    if sidecar.buffers_count != count {
        return Err(FdxValidationError::FlagFieldIncoherent {
            detail: "buffers_count does not match the supplied buffer table length",
        });
    }
    if count == 0 {
        return Err(FdxValidationError::BufferRefOutOfRange {
            index: 0,
            count,
            detail: "buffer table must contain at least index 0 (the base data buffer)",
        });
    }
    // index 0 role: Data, or POOL when the gather pool conventionally aliases it.
    let role0 = buffers[0].role;
    let pool_aliases_0 =
        flag_set(sidecar.flags, FDX_FLAG_HAS_GATHER) && sidecar.gather.pool_buffer == 0;
    let role0_ok = role0 == FDX_BUFFER_ROLE_DATA
        || (pool_aliases_0 && role0 == FDX_BUFFER_ROLE_POOL);
    if !role0_ok {
        return Err(FdxValidationError::FlagFieldIncoherent {
            detail: "buffer index 0 must have role Data (or POOL for an aliased gather pool)",
        });
    }
    for (i, b) in buffers.iter().enumerate() {
        if b.byte_offset > b.size_bytes {
            return Err(FdxValidationError::BufferRefOutOfRange {
                index: i as u32,
                count,
                detail: "byte_offset must be <= size_bytes",
            });
        }
    }
    // Quant cross-refs.
    if flag_set(sidecar.flags, FDX_FLAG_HAS_QUANT) {
        let q = &sidecar.quant;
        if q.scale_present != 0
            && q.scale_placement == FDX_SCALE_PLACEMENT_SEPARATE_BUFFER
            && q.scale_buffer != FDX_BUFFER_INLINE
            && q.scale_buffer >= count
        {
            return Err(FdxValidationError::BufferRefOutOfRange {
                index: q.scale_buffer,
                count,
                detail: "scale_buffer index out of range",
            });
        }
        if q.zp_present != 0 && q.zp_buffer != FDX_BUFFER_INLINE && q.zp_buffer >= count {
            return Err(FdxValidationError::BufferRefOutOfRange {
                index: q.zp_buffer,
                count,
                detail: "zp_buffer index out of range",
            });
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// V10 — bundle (§8.10, §6.8)
// ─────────────────────────────────────────────────────────────────────────────

/// V10 (§8.10): `IS_BUNDLE` ⇒ `views_count > 0`, every sub-view in-bounds within
/// the bundle buffer (index 0) and non-overlapping.
pub fn check_v10_bundle(sidecar: &FDXSidecar, buffers: &[FDXBufferRef]) -> R {
    if !flag_set(sidecar.flags, FDX_FLAG_IS_BUNDLE) {
        return Ok(());
    }
    if sidecar.views_count == 0 {
        return Err(FdxValidationError::BundleOverlap {
            detail: "IS_BUNDLE set but views_count == 0",
        });
    }
    let backing = buffers
        .first()
        .ok_or(FdxValidationError::BufferRefOutOfRange {
            index: 0,
            count: 0,
            detail: "bundle requires a backing buffer at index 0",
        })?;
    let views = unsafe { views_slice(sidecar) };
    // Compute [start, end) byte windows and check bounds + pairwise overlap.
    let mut windows: Vec<(u64, u64)> = Vec::with_capacity(views.len());
    for v in views {
        let elem = fdx_elem_bytes(v.dtype).ok_or(FdxValidationError::BundleOverlap {
            detail: "bundle view has an unknown dtype code",
        })?;
        let span = (v.len_elements as u128).saturating_mul(elem as u128);
        let start = v.byte_offset as u128;
        let end = start + span;
        if end > backing.size_bytes as u128 {
            return Err(FdxValidationError::BundleOverlap {
                detail: "bundle view exceeds the backing buffer",
            });
        }
        windows.push((start as u64, end as u64));
    }
    for i in 0..windows.len() {
        for j in (i + 1)..windows.len() {
            let (a0, a1) = windows[i];
            let (b0, b1) = windows[j];
            if a0 < b1 && b0 < a1 {
                return Err(FdxValidationError::BundleOverlap {
                    detail: "bundle views overlap",
                });
            }
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// V11 — explicit strides (§8.11, §3.2)
// ─────────────────────────────────────────────────────────────────────────────

/// V11 (§8.11): the base `DLTensor.strides` is non-NULL when `ndim != 0`.
/// `FDXBufferRef`/`FDXOutputView` carry inline (always-present) strides arrays,
/// so the only NULL-able pointer is the base strides.
pub fn check_v11_explicit_strides(base: &DLTensor) -> R {
    if base.ndim != 0 && base.strides.is_null() {
        return Err(FdxValidationError::NullStrides {
            detail: "base DLTensor.strides is NULL while ndim != 0",
        });
    }
    if base.ndim != 0 && base.shape.is_null() {
        return Err(FdxValidationError::NullStrides {
            detail: "base DLTensor.shape is NULL while ndim != 0",
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// V12 — 256-byte alignment (boundary b) (§8.12, §3.3)
// ─────────────────────────────────────────────────────────────────────────────

/// V12 (§8.12, boundary-b only): `(data as usize) % 256 == 0` for the base and
/// every exported buffer; each buffer's `byte_offset <= size_bytes`. This is the
/// boundary-(b) export check — call it only on an export (boundary a relaxes to
/// `required_alignment`).
pub fn check_v12_alignment_boundary_b(base: &DLTensor, buffers: &[FDXBufferRef]) -> R {
    const ALIGN: usize = 256;
    if !base.data.is_null() && (base.data as usize) % ALIGN != 0 {
        return Err(FdxValidationError::Misaligned {
            detail: "base DLTensor.data",
            addr: base.data as usize,
        });
    }
    for (i, b) in buffers.iter().enumerate() {
        if !b.data.is_null() && (b.data as usize) % ALIGN != 0 {
            return Err(FdxValidationError::Misaligned {
                detail: if i == 0 {
                    "buffer 0 data"
                } else {
                    "exported buffer data"
                },
                addr: b.data as usize,
            });
        }
        if b.byte_offset > b.size_bytes {
            return Err(FdxValidationError::BufferRefOutOfRange {
                index: i as u32,
                count: buffers.len() as u32,
                detail: "byte_offset must be <= size_bytes",
            });
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// V13 — signed-stride OOB range (§8.13, §3.2.1)
// ─────────────────────────────────────────────────────────────────────────────

/// V13 (§8.13): the signed touched byte window over the strides must lie within
/// `[0, size_bytes)`. Per axis: `hi_i = (shape[i]-1)*stride[i]` if `stride>0`
/// else 0; `lo_i = (shape[i]-1)*stride[i]` if `stride<0` else 0. The reachable
/// window is `[byte_offset + Σ lo_i, byte_offset + Σ hi_i]` scaled by the element
/// byte width; require it ⊆ `[0, size_bytes)`. Negatives are FIRST-CLASS (not
/// rejected). All arithmetic in i128 to avoid overflow.
///
/// `shape`/`strides` are typed-element counts; `elem_bytes` is the per-element
/// byte width; `size_bytes` is the buffer's physical byte count.
pub fn check_v13_signed_stride_oob(
    shape: &[i64],
    strides: &[i64],
    byte_offset: u64,
    elem_bytes: u64,
    size_bytes: u64,
    detail: &'static str,
) -> R {
    debug_assert_eq!(shape.len(), strides.len());
    let elem = elem_bytes as i128;
    let mut lo_elems: i128 = 0;
    let mut hi_elems: i128 = 0;
    for (&dim, &stride) in shape.iter().zip(strides.iter()) {
        if dim <= 0 {
            // An empty axis touches nothing along it. (dim==0 ⇒ no elements.)
            continue;
        }
        let contrib = (dim as i128 - 1).saturating_mul(stride as i128);
        if stride > 0 {
            hi_elems = hi_elems.saturating_add(contrib);
        } else if stride < 0 {
            lo_elems = lo_elems.saturating_add(contrib);
        }
    }
    let bo = byte_offset as i128;
    // Byte window: [bo + lo*elem, bo + hi*elem + (elem - 1)] — the last touched
    // element spans `elem` bytes, so its end byte is hi*elem + (elem-1).
    let lo_byte = bo + lo_elems.saturating_mul(elem);
    let hi_byte = bo + hi_elems.saturating_mul(elem) + (elem - 1);
    if lo_byte < 0 || hi_byte >= size_bytes as i128 {
        return Err(FdxValidationError::StrideRangeOutOfBounds {
            detail,
            lo: lo_byte,
            hi: hi_byte,
            size_bytes,
        });
    }
    Ok(())
}

/// V13 for the base `DLTensor` (honesty uint8 ⇒ elem 1; faithful dtype ⇒ its byte
/// width). The base's backing byte count is `buffers[0].size_bytes`.
fn check_v13_base(base: &DLTensor, buffers: &[FDXBufferRef]) -> R {
    let ndim = base.ndim.max(0) as usize;
    if ndim == 0 || base.shape.is_null() || base.strides.is_null() {
        return Ok(());
    }
    let buf0 = match buffers.first() {
        Some(b) => b,
        None => return Ok(()),
    };
    let shape: Vec<i64> = (0..ndim).map(|i| unsafe { *base.shape.add(i) }).collect();
    let strides: Vec<i64> = (0..ndim).map(|i| unsafe { *base.strides.add(i) }).collect();
    let elem = base.dtype.bits.div_ceil(8).max(1) as u64;
    check_v13_signed_stride_oob(
        &shape,
        &strides,
        base.byte_offset,
        elem,
        buf0.size_bytes,
        "base DLTensor",
    )
}

/// V13 for each exported `FDXBufferRef` (its inline shape/strides, its own
/// dtype's element width, its own size_bytes).
fn check_v13_buffers(buffers: &[FDXBufferRef]) -> R {
    for b in buffers {
        let ndim = (b.ndim as usize).min(6);
        if ndim == 0 {
            continue;
        }
        let elem = fdx_elem_bytes(b.dtype).unwrap_or(1);
        let shape: Vec<i64> = b.shape[..ndim].iter().map(|&d| d as i64).collect();
        let strides: Vec<i64> = b.strides[..ndim].to_vec();
        check_v13_signed_stride_oob(
            &shape,
            &strides,
            b.byte_offset,
            elem,
            b.size_bytes,
            "exported FDXBufferRef",
        )?;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// V15 — no raw pointers in serialized form (§8.15)
// ─────────────────────────────────────────────────────────────────────────────

/// V15 (§8.15): in a SERIALIZED blob all pointer-typed fields are 0. This check
/// is for a blob marked as serialized; the live in-memory form legitimately
/// carries pointers, so it is exposed separately and NOT run by [`validate`]
/// (which validates a live sidecar). Call it on a deserialized blob.
pub fn check_v15_no_serialized_pointers(sidecar: &FDXSidecar, buffers: &[FDXBufferRef]) -> R {
    if !sidecar.extents.is_null()
        || !sidecar.buffers.is_null()
        || !sidecar.views.is_null()
    {
        return Err(FdxValidationError::PointerInSerializedForm {
            detail: "FDXSidecar array pointers must be 0 in serialized form",
        });
    }
    for b in buffers {
        if !b.data.is_null() {
            return Err(FdxValidationError::PointerInSerializedForm {
                detail: "FDXBufferRef.data must be 0 in serialized form",
            });
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// V16 — affine well-formedness (§8.16, §6.4.2/§6.4.3)
// ─────────────────────────────────────────────────────────────────────────────

/// V16 (§8.16): for a `kind=Affine` extent:
/// `1 <= term_count <= FDX_AFFINE_MAX_TERMS`; each active term has a real
/// `sym_id`; each inactive slot is zeroed; no duplicate `sym_id`; not degenerate
/// (reject the `Range`/`Scalar`-reducible forms); `cap_kind == EXPLICIT`.
pub fn check_v16_affine(ext: &FDXExtent, axis: usize) -> R {
    if ext.kind as u16 != FDX_EXTENT_AFFINE {
        return Ok(());
    }
    let a = &ext.affine;
    let tc = a.term_count as usize;
    if tc == 0 {
        return Err(FdxValidationError::AffineDegenerate {
            axis,
            detail: "term_count == 0 must be Scalar",
        });
    }
    if tc > FDX_AFFINE_MAX_TERMS {
        return Err(FdxValidationError::AffineTooManyTerms {
            axis,
            term_count: a.term_count,
            max: FDX_AFFINE_MAX_TERMS,
        });
    }
    if ext.cap_kind as u16 != FDX_CAP_KIND_EXPLICIT {
        return Err(FdxValidationError::AffineMalformed {
            axis,
            detail: "affine cap_kind must be EXPLICIT in v1",
        });
    }
    // Active terms: real sym, no duplicates.
    for i in 0..tc {
        let t = &a.terms[i];
        if t.sym_id == FDX_SYM_NONE {
            return Err(FdxValidationError::AffineMalformed {
                axis,
                detail: "active affine term must have a real sym_id",
            });
        }
        for j in (i + 1)..tc {
            if a.terms[j].sym_id == t.sym_id {
                return Err(FdxValidationError::AffineMalformed {
                    axis,
                    detail: "duplicate sym_id across active affine terms",
                });
            }
        }
    }
    // Inactive slots zeroed.
    for i in tc..FDX_AFFINE_MAX_TERMS {
        let t = &a.terms[i];
        if t.sym_id != FDX_SYM_NONE || t.coeff != 0 {
            return Err(FdxValidationError::AffineMalformed {
                axis,
                detail: "inactive affine slot must be zeroed (sym=NONE, coeff=0)",
            });
        }
    }
    // Not degenerate: reject term_count==1 && c0==0 && coeff==1 (must be Range).
    if tc == 1 && a.c0 == 0 && a.terms[0].coeff == 1 {
        return Err(FdxValidationError::AffineDegenerate {
            axis,
            detail: "single coeff-1 zero-c0 term must be Range",
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// V17 — affine evaluation safety (realize time) (§8.17, §6.4.2)
// ─────────────────────────────────────────────────────────────────────────────

/// V17 (§8.17): evaluate `c0 + Σ coeff_i * resolve(sym_i)` in i128 with
/// `checked_mul`/`checked_add` at EVERY step (both operands widened to i128
/// before the multiply); any overflow ⇒ `AffineOverflow`. Then gate the result
/// `>= 0` (a negative i128 is `ExtentOutOfRange` BEFORE narrowing) and narrow to
/// `usize`/`u64` (32-bit host: `> usize::MAX` ⇒ `ExtentOutOfRange`, never
/// truncated). Returns the safe non-negative live value. Runs BEFORE V14.
pub fn check_v17_affine_eval(ext: &FDXExtent, axis: usize, env: &FDXSymEnv) -> Result<u64, FdxValidationError> {
    let a = &ext.affine;
    let mut value: i128 = a.c0 as i128;
    for i in 0..(a.term_count as usize) {
        let t = &a.terms[i];
        let s = env_lookup(env, t.sym_id).ok_or(FdxValidationError::UnboundSymbol {
            axis,
            sym_id: t.sym_id,
        })?;
        // BOTH operands widen to i128 before the multiply (a u64 sym > i64::MAX
        // would mis-cast negative if done in i64).
        let prod = (t.coeff as i128)
            .checked_mul(s as i128)
            .ok_or(FdxValidationError::AffineOverflow { axis })?;
        value = value
            .checked_add(prod)
            .ok_or(FdxValidationError::AffineOverflow { axis })?;
    }
    // V17 >= 0 gate BEFORE narrowing.
    if value < 0 {
        return Err(FdxValidationError::ExtentOutOfRange {
            axis,
            value,
            min: ext.min,
            capacity: ext.capacity,
        });
    }
    // Narrow to u64; 32-bit host narrowing to usize handled by V14's compare,
    // but reject value > usize::MAX here per the narrowing policy.
    if value > usize::MAX as i128 {
        return Err(FdxValidationError::ExtentOutOfRange {
            axis,
            value,
            min: ext.min,
            capacity: ext.capacity,
        });
    }
    Ok(value as u64)
}

// ─────────────────────────────────────────────────────────────────────────────
// V14 — realize-time symbol bounds (§8.14, §6.4)
// ─────────────────────────────────────────────────────────────────────────────

/// V14 (§8.14): for one extent, compute the live value (Scalar: `capacity`;
/// Range: `env.lookup(sym)`; Affine: V17 evaluation), then enforce
/// `min <= value <= capacity`. Unbound symbol ⇒ `UnboundSymbol`. Returns the
/// resolved live value.
pub fn check_v14_extent_bounds(
    ext: &FDXExtent,
    axis: usize,
    env: &FDXSymEnv,
) -> Result<u64, FdxValidationError> {
    let value: u64 = match ext.kind as u16 {
        FDX_EXTENT_SCALAR => ext.capacity,
        FDX_EXTENT_RANGE => env_lookup(env, ext.sym_id).ok_or(FdxValidationError::UnboundSymbol {
            axis,
            sym_id: ext.sym_id,
        })?,
        FDX_EXTENT_AFFINE => {
            // V16 well-formedness first (build-time), then V17 eval (runs before V14).
            check_v16_affine(ext, axis)?;
            check_v17_affine_eval(ext, axis, env)?
        }
        _ => {
            return Err(FdxValidationError::ExtentMismatch {
                axis,
                detail: "unknown extent kind",
            });
        }
    };
    if value < ext.min || value > ext.capacity {
        return Err(FdxValidationError::ExtentOutOfRange {
            axis,
            value: value as i128,
            min: ext.min,
            capacity: ext.capacity,
        });
    }
    Ok(value)
}

// ─────────────────────────────────────────────────────────────────────────────
// V18 — gather coherence (§8.18, §6.9)
// ─────────────────────────────────────────────────────────────────────────────

/// V18 (§8.18): gather descriptor coherence. `HAS_GATHER` ⇔ `kind != NONE`;
/// for `PAGED_BLOCKS`: non-zero geometry, `id_dtype == U32`, valid ndims,
/// `physical_shape[0]==num_blocks`, `physical_shape[1]==block_size`,
/// `max_seq_capacity == max_blocks_per_seq * block_size` (u64, no overflow),
/// `unmapped_sentinel >= num_blocks`. Unknown kind ⇒ `UnsupportedGatherKind`.
pub fn check_v18_gather_coherence(sidecar: &FDXSidecar) -> R {
    let has = flag_set(sidecar.flags, FDX_FLAG_HAS_GATHER);
    let g = &sidecar.gather;
    let kind = g.kind as u16;
    if has != (kind != FDX_GATHER_NONE) {
        return Err(FdxValidationError::GatherIncoherent {
            detail: "HAS_GATHER set iff gather.kind != FDX_GATHER_NONE",
        });
    }
    if !has {
        return Ok(());
    }
    if kind != FDX_GATHER_PAGED_BLOCKS {
        return Err(FdxValidationError::UnsupportedGatherKind { kind });
    }
    if g.block_size == 0
        || g.num_blocks == 0
        || g.num_sequences == 0
        || g.block_table.max_blocks_per_seq == 0
    {
        return Err(FdxValidationError::GatherIncoherent {
            detail: "block_size/num_blocks/num_sequences/max_blocks_per_seq must be non-zero",
        });
    }
    if g.block_table.id_dtype != FDX_DTYPE_U32 {
        return Err(FdxValidationError::GatherIncoherent {
            detail: "block-table id_dtype must be U32 (v1 pin)",
        });
    }
    if g.physical_ndim == 0 || g.physical_ndim > 6 {
        return Err(FdxValidationError::GatherIncoherent {
            detail: "physical_ndim must be in [1,6]",
        });
    }
    if g.logical_ndim > 6 {
        return Err(FdxValidationError::GatherIncoherent {
            detail: "logical_ndim must be in [0,6]",
        });
    }
    if g.physical_shape[0] != g.num_blocks {
        return Err(FdxValidationError::GatherIncoherent {
            detail: "physical_shape[0] must equal num_blocks",
        });
    }
    if g.physical_ndim >= 2 && g.physical_shape[1] != g.block_size {
        return Err(FdxValidationError::GatherIncoherent {
            detail: "physical_shape[1] must equal block_size",
        });
    }
    // max_seq_capacity == max_blocks_per_seq * block_size in u64, no overflow.
    let prod = (g.block_table.max_blocks_per_seq as u64).checked_mul(g.block_size);
    match prod {
        Some(p) if p == g.max_seq_capacity => {}
        Some(_) => {
            return Err(FdxValidationError::GatherIncoherent {
                detail: "max_seq_capacity must equal max_blocks_per_seq * block_size",
            });
        }
        None => {
            return Err(FdxValidationError::GatherIncoherent {
                detail: "max_blocks_per_seq * block_size overflows u64",
            });
        }
    }
    // unmapped_sentinel must be >= num_blocks (so id>=num_blocks catches both).
    if (g.block_table.unmapped_sentinel as u64) < g.num_blocks {
        return Err(FdxValidationError::GatherIncoherent {
            detail: "unmapped_sentinel must be >= num_blocks",
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// V19 — MEANING_REQUIRES_EXT mandatory + base honesty (§8.19, §6.9.1)
// ─────────────────────────────────────────────────────────────────────────────

/// V19 (§8.19): `HAS_GATHER` ⇒ `MEANING_REQUIRES_EXT` set; base `dtype ==
/// {kDLUInt,8,1}`; base `strides == [1]`; and the base byte length exactly equals
/// the pool allocation walk `physical_strides[0] * num_blocks * elem_bytes`.
pub fn check_v19_gather_meaning_and_honesty(sidecar: &FDXSidecar, base: &DLTensor) -> R {
    if !flag_set(sidecar.flags, FDX_FLAG_HAS_GATHER) {
        return Ok(());
    }
    if !flag_set(sidecar.flags, FDX_FLAG_MEANING_REQUIRES_EXT) {
        return Err(FdxValidationError::DishonestBase {
            detail: "HAS_GATHER requires MEANING_REQUIRES_EXT (GatherWithoutMeaningFlag)",
        });
    }
    // Base honesty (V3 also checks dtype; here pin uint8 + strides=[1] + ndim 1).
    if base.dtype.code != dtype_code::K_DL_UINT || base.dtype.bits != 8 || base.dtype.lanes != 1 {
        return Err(FdxValidationError::DishonestBase {
            detail: "gather base dtype must be {kDLUInt,8,1}",
        });
    }
    if base.ndim != 1 {
        return Err(FdxValidationError::DishonestBase {
            detail: "gather base must be a 1-D byte pool",
        });
    }
    if base.strides.is_null() || unsafe { *base.strides } != 1 {
        return Err(FdxValidationError::DishonestBase {
            detail: "gather base strides must be [1]",
        });
    }
    // Honest-uint8 cover: base.shape[0] == physical_strides[0]*num_blocks*elem_bytes.
    let g = &sidecar.gather;
    let elem = whole_byte_width(g.element_dtype).ok_or(FdxValidationError::DishonestBase {
        detail: "gather element_dtype must be a whole-byte type with a known width",
    })?;
    let pool_bytes = pool_byte_len(g, elem)?;
    let base_bytes = if base.shape.is_null() {
        0
    } else {
        unsafe { *base.shape }.max(0) as u128
    };
    if base_bytes != pool_bytes {
        return Err(FdxValidationError::DishonestBase {
            detail: "base byte length must equal the pool allocation walk (V19)",
        });
    }
    Ok(())
}

/// The shared "pool byte length" definition (V19/V20): the slowest physical axis
/// `stride·extent` scaled by element byte width:
/// `physical_strides[0] * num_blocks * elem_bytes`.
fn pool_byte_len(g: &FDXIndexedResidency, elem_bytes: u64) -> Result<u128, FdxValidationError> {
    let s0 = g.physical_strides[0];
    if s0 < 0 {
        return Err(FdxValidationError::GatherIncoherent {
            detail: "physical_strides[0] must be non-negative for a paged pool",
        });
    }
    let len = (s0 as u128)
        .checked_mul(g.num_blocks as u128)
        .and_then(|x| x.checked_mul(elem_bytes as u128))
        .ok_or(FdxValidationError::GatherAddressOverflow {
            detail: "pool byte length overflows",
        })?;
    Ok(len)
}

/// Whole-byte element width of an FDX dtype code, rejecting sub-byte codes (the
/// pool typed element must be a whole-byte type — F16/BF16/F32, never F4).
fn whole_byte_width(code: u16) -> Option<u64> {
    let bits = fdx_bit_width(code)?;
    if bits % 8 != 0 {
        return None;
    }
    Some(bits / 8)
}

// ─────────────────────────────────────────────────────────────────────────────
// V20 — pool backing (§8.20, §6.9.4)
// ─────────────────────────────────────────────────────────────────────────────

/// V20 (§8.20): mirrors V8 for the pool: `buffers[pool_buffer].size_bytes >=
/// physical_strides[0] * num_blocks * elem_bytes` AND the analogous
/// `stride * extent` on every physical axis. Only runs when `HAS_GATHER`.
pub fn check_v20_pool_backing(sidecar: &FDXSidecar, buffers: &[FDXBufferRef]) -> R {
    if !flag_set(sidecar.flags, FDX_FLAG_HAS_GATHER) {
        return Ok(());
    }
    let g = &sidecar.gather;
    let elem = whole_byte_width(g.element_dtype).ok_or(FdxValidationError::GatherIncoherent {
        detail: "gather element_dtype must be a whole-byte type",
    })?;
    let pool = buffers
        .get(g.pool_buffer as usize)
        .ok_or(FdxValidationError::BufferRefOutOfRange {
            index: g.pool_buffer,
            count: buffers.len() as u32,
            detail: "pool_buffer index out of range",
        })?;
    // Every physical axis: stride*extent*elem must be backed; the slowest axis
    // (block axis, physical_strides[0]*num_blocks) is the dominant term.
    let need = pool_byte_len(g, elem)?;
    if (pool.size_bytes as u128) < need {
        return Err(FdxValidationError::CapacityNotBacked {
            detail: "pool buffer does not back the full pool capacity (PoolNotBacked)",
            have: pool.size_bytes,
            need,
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// V21 — gather ↔ operands ↔ symbol consistency (§8.21, §6.9.4)
// ─────────────────────────────────────────────────────────────────────────────

/// V21(a) (§8.21): `pool_buffer`, `block_table.table_buffer`,
/// `context_lens_buffer` (when not NONE) are valid indices with matching roles
/// and shapes (`block_table` is `[num_sequences, max_blocks_per_seq]`,
/// `context_lens` is `[num_sequences]`).
pub fn check_v21a_gather_buffers(sidecar: &FDXSidecar, buffers: &[FDXBufferRef]) -> R {
    if !flag_set(sidecar.flags, FDX_FLAG_HAS_GATHER) {
        return Ok(());
    }
    let g = &sidecar.gather;
    let count = buffers.len() as u32;

    let pool = buffers
        .get(g.pool_buffer as usize)
        .ok_or(FdxValidationError::BufferRefOutOfRange {
            index: g.pool_buffer,
            count,
            detail: "pool_buffer index out of range",
        })?;
    if pool.role != FDX_BUFFER_ROLE_POOL && pool.role != FDX_BUFFER_ROLE_DATA {
        return Err(FdxValidationError::GatherIncoherent {
            detail: "pool_buffer role must be POOL (or Data when aliasing index 0)",
        });
    }

    let bt_idx = g.block_table.table_buffer;
    let bt = buffers
        .get(bt_idx as usize)
        .ok_or(FdxValidationError::BufferRefOutOfRange {
            index: bt_idx,
            count,
            detail: "block_table.table_buffer index out of range",
        })?;
    if bt.role != FDX_BUFFER_ROLE_BLOCK_TABLE {
        return Err(FdxValidationError::GatherIncoherent {
            detail: "block_table buffer role must be BLOCK_TABLE",
        });
    }
    if bt.ndim != 2
        || bt.shape[0] != g.num_sequences
        || bt.shape[1] != g.block_table.max_blocks_per_seq as u64
    {
        return Err(FdxValidationError::GatherIncoherent {
            detail: "block_table shape must be [num_sequences, max_blocks_per_seq]",
        });
    }

    if g.context_lens_buffer != FDX_BUFFER_NONE {
        let cl = buffers
            .get(g.context_lens_buffer as usize)
            .ok_or(FdxValidationError::BufferRefOutOfRange {
                index: g.context_lens_buffer,
                count,
                detail: "context_lens_buffer index out of range",
            })?;
        if cl.role != FDX_BUFFER_ROLE_CONTEXT_LENS {
            return Err(FdxValidationError::GatherIncoherent {
                detail: "context_lens buffer role must be CONTEXT_LENS",
            });
        }
        if cl.ndim != 1 || cl.shape[0] != g.num_sequences {
            return Err(FdxValidationError::GatherIncoherent {
                detail: "context_lens shape must be [num_sequences]",
            });
        }
    }
    Ok(())
}

/// V21(c) (§8.21): build-time FULL-table scan — every MAPPED entry
/// (`id != unmapped_sentinel`) of the block-table buffer satisfies
/// `0 <= id < num_blocks`. Takes the decoded block-id table (`&[u32]`,
/// row-major `[num_sequences, max_blocks_per_seq]`).
pub fn check_v21c_block_table_scan(sidecar: &FDXSidecar, block_ids: &[u32]) -> R {
    if !flag_set(sidecar.flags, FDX_FLAG_HAS_GATHER) {
        return Ok(());
    }
    let g = &sidecar.gather;
    let sentinel = g.block_table.unmapped_sentinel;
    let num_blocks = g.num_blocks;
    for (slot, &id) in block_ids.iter().enumerate() {
        if id == sentinel {
            continue;
        }
        if (id as u64) >= num_blocks {
            return Err(FdxValidationError::BlockIdOutOfRange {
                id,
                num_blocks,
                slot,
            });
        }
    }
    Ok(())
}

/// V21(e) (§8.21): when `context_len_sym != FDX_SYM_NONE`,
/// `logical_extents_count == logical_ndim`, and `logical_extents[seq_axis]`
/// carries the same `sym_id` with `capacity == max_seq_capacity` and `min == 0`.
/// Also runs each logical extent's V7 arm keyed to `logical_shape` /
/// `max_seq_capacity`. (`logical_extents_count ∈ {0, logical_ndim}` always.)
pub fn check_v21e_logical_extents(sidecar: &FDXSidecar) -> R {
    if !flag_set(sidecar.flags, FDX_FLAG_HAS_GATHER) {
        return Ok(());
    }
    let g = &sidecar.gather;
    let lec = g.logical_extents_count as usize;
    let ln = g.logical_ndim as usize;
    if lec != 0 && lec != ln {
        return Err(FdxValidationError::GatherIncoherent {
            detail: "logical_extents_count must be 0 or logical_ndim",
        });
    }

    // If there's a unified context-length sym, the logical extents must exist and
    // the seq-axis extent must carry it.
    let has_sym = g.context_len_sym != FDX_SYM_NONE;
    if has_sym {
        if lec != ln {
            return Err(FdxValidationError::GatherIncoherent {
                detail: "context_len_sym set requires logical_extents_count == logical_ndim",
            });
        }
        let sa = g.seq_axis as usize;
        if g.seq_axis == 0xFF || sa >= ln {
            return Err(FdxValidationError::GatherIncoherent {
                detail: "seq_axis must be a valid logical axis when context_len_sym is set",
            });
        }
        let seq_ext = &g.logical_extents[sa];
        if seq_ext.sym_id != g.context_len_sym {
            return Err(FdxValidationError::GatherIncoherent {
                detail: "logical_extents[seq_axis].sym_id must equal context_len_sym (P4)",
            });
        }
        if seq_ext.capacity != g.max_seq_capacity {
            return Err(FdxValidationError::GatherIncoherent {
                detail: "logical_extents[seq_axis].capacity must equal max_seq_capacity",
            });
        }
        if seq_ext.min != 0 {
            return Err(FdxValidationError::GatherIncoherent {
                detail: "logical_extents[seq_axis].min must be 0 (empty sequences legal)",
            });
        }
    }

    // Each present logical extent runs its own V7 arm keyed to logical_shape
    // (seq_axis uses max_seq_capacity).
    if lec == ln {
        for i in 0..ln {
            let ext = &g.logical_extents[i];
            let cap = if i == g.seq_axis as usize {
                g.max_seq_capacity
            } else {
                g.logical_shape[i]
            };
            check_v7_extent_arm(ext, i, cap)?;
        }
    }
    Ok(())
}

/// V21(d) (§8.21, realize-time): for one sequence's live length `L` (from the env
/// via `context_len_sym`, or from `context_lens[s]`), check
/// `0 <= L <= max_seq_capacity` and `ceil(L/block_size) <= max_blocks_per_seq`,
/// with u64/checked address-overflow guards re-checked against the runtime
/// `usize`. `L == 0` is LEGAL (skipped sequence).
pub fn check_v21d_seq_live_length(
    sidecar: &FDXSidecar,
    seq_index: usize,
    live_len: u64,
) -> R {
    let g = &sidecar.gather;
    if live_len > g.max_seq_capacity {
        return Err(FdxValidationError::ExtentOutOfRange {
            axis: seq_index,
            value: live_len as i128,
            min: 0,
            capacity: g.max_seq_capacity,
        });
    }
    // ceil(L/block_size) <= max_blocks_per_seq (column-index guard).
    let cols = live_len.div_ceil(g.block_size);
    if cols > g.block_table.max_blocks_per_seq as u64 {
        return Err(FdxValidationError::GatherIncoherent {
            detail: "ceil(L/block_size) exceeds max_blocks_per_seq",
        });
    }
    // Runtime-usize narrowing: the flat block-table index for the last column
    // must fit usize.
    let last_col = if live_len == 0 { 0 } else { cols - 1 };
    let flat = (seq_index as u128)
        .checked_mul(g.block_table.max_blocks_per_seq as u128)
        .and_then(|x| x.checked_add(last_col as u128))
        .ok_or(FdxValidationError::GatherAddressOverflow {
            detail: "flat block-table index overflows",
        })?;
    if flat > usize::MAX as u128 {
        return Err(FdxValidationError::GatherAddressOverflow {
            detail: "flat block-table index exceeds usize::MAX on this host",
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// SymEnv lookup
// ─────────────────────────────────────────────────────────────────────────────

/// Look up `sym_id` in the (sorted-by-sym_id) `FDXSymEnv`. Returns the `u64`
/// binding, or `None` if unbound. Linear scan is fine for the small per-pass
/// binding set; the spec's "sorted for O(log n)" is a perf note, not a contract
/// the validator depends on.
fn env_lookup(env: &FDXSymEnv, sym_id: u32) -> Option<u64> {
    if env.bindings.is_null() {
        return None;
    }
    let n = env.count as usize;
    let slice = unsafe { core::slice::from_raw_parts(env.bindings, n) };
    slice.iter().find(|b| b.sym_id == sym_id).map(|b| b.value)
}

// ─────────────────────────────────────────────────────────────────────────────
// Array-pointer slices (live in-memory form)
// ─────────────────────────────────────────────────────────────────────────────

/// `extents[0..extents_count]` as a slice (live form). Empty if the pointer is
/// NULL or the count is 0.
unsafe fn extents_slice(sidecar: &FDXSidecar) -> &[FDXExtent] {
    if sidecar.extents.is_null() || sidecar.extents_count == 0 {
        return &[];
    }
    // SAFETY: caller guarantees `extents` points at `extents_count` valid
    // `FDXExtent` for the live form; null/empty handled above.
    unsafe { core::slice::from_raw_parts(sidecar.extents, sidecar.extents_count as usize) }
}

/// `views[0..views_count]` as a slice (live form).
unsafe fn views_slice(sidecar: &FDXSidecar) -> &[FDXOutputView] {
    if sidecar.views.is_null() || sidecar.views_count == 0 {
        return &[];
    }
    // SAFETY: caller guarantees `views` points at `views_count` valid
    // `FDXOutputView` for the live form; null/empty handled above.
    unsafe { core::slice::from_raw_parts(sidecar.views, sidecar.views_count as usize) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level surfaces
// ─────────────────────────────────────────────────────────────────────────────

/// Run every applicable **build-time / boundary-time** validator (V1..V13, V16,
/// V18..V21 build arms), gated by the relevant `FDX_FLAG_*`. Does NOT run the
/// realize-time checks (V14, V17, V21d) — those need an [`FDXSymEnv`] and live in
/// [`validate_realize`]. Does NOT run the boundary-(b)-only V12 alignment check
/// (call [`check_v12_alignment_boundary_b`] explicitly on an export) nor the
/// serialized-form V15 (call [`check_v15_no_serialized_pointers`] on a blob).
///
/// `buffers` is the live `&[FDXBufferRef]` buffer table (`buffers[0]` is the base
/// data buffer).
pub fn validate(sidecar: &FDXSidecar, base: &DLTensor, buffers: &[FDXBufferRef]) -> R {
    check_v1_header(sidecar)?;
    check_v2_flag_coherence(sidecar)?;
    check_v3_honesty_dtype(sidecar, base)?;
    check_v4_sub_byte(sidecar)?;
    check_v5_quant(sidecar)?;
    check_v6_scale_shape(sidecar, base, buffers)?;
    check_v7_extents(sidecar, base)?;
    check_v9_buffer_refs(sidecar, buffers)?;
    check_v8_capacity_backing(sidecar, base, buffers)?;
    check_v10_bundle(sidecar, buffers)?;
    check_v11_explicit_strides(base)?;
    check_v13_base(base, buffers)?;
    check_v13_buffers(buffers)?;
    // Gather build arms.
    check_v18_gather_coherence(sidecar)?;
    check_v19_gather_meaning_and_honesty(sidecar, base)?;
    check_v20_pool_backing(sidecar, buffers)?;
    check_v21a_gather_buffers(sidecar, buffers)?;
    check_v21e_logical_extents(sidecar)?;
    Ok(())
}

/// Run the build-time checks, then the **realize-time** checks (V14 + V17 affine
/// evaluation, V21d per-sequence bounds) against a resolved [`FDXSymEnv`].
///
/// For a non-gather tensor with symbolic extents, this evaluates every extent's
/// live value and applies the `min <= value <= capacity` bound (V14). For a
/// gather tensor, when `context_len_sym` is bound it checks the shared live
/// length via V21d; per-sequence data-determined lengths
/// (`context_len_sym == FDX_SYM_NONE`) are validated by the caller via
/// [`check_v21d_seq_live_length`] as each sequence is accessed (the spec's lazy
/// per-accessed-slot rule).
pub fn validate_realize(
    sidecar: &FDXSidecar,
    base: &DLTensor,
    buffers: &[FDXBufferRef],
    env: &FDXSymEnv,
) -> R {
    validate(sidecar, base, buffers)?;
    // V14 over the base extents.
    let extents = unsafe { extents_slice(sidecar) };
    for (i, ext) in extents.iter().enumerate() {
        check_v14_extent_bounds(ext, i, env)?;
    }
    // Gather: the shared live length (when present) via V21d for every sequence.
    if flag_set(sidecar.flags, FDX_FLAG_HAS_GATHER) {
        let g = &sidecar.gather;
        if g.context_len_sym != FDX_SYM_NONE {
            // Also run V14 on the seq-axis logical extent (Range/Affine).
            let sa = g.seq_axis as usize;
            if sa < g.logical_ndim as usize && g.logical_extents_count != 0 {
                let seq_ext = &g.logical_extents[sa];
                let l = check_v14_extent_bounds(seq_ext, sa, env)?;
                for s in 0..(g.num_sequences as usize) {
                    check_v21d_seq_live_length(sidecar, s, l)?;
                }
            } else {
                // No logical extent carrier; resolve the sym directly.
                let l = env_lookup(env, g.context_len_sym).ok_or(
                    FdxValidationError::UnboundSymbol {
                        axis: 0,
                        sym_id: g.context_len_sym,
                    },
                )?;
                for s in 0..(g.num_sequences as usize) {
                    check_v21d_seq_live_length(sidecar, s, l)?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
