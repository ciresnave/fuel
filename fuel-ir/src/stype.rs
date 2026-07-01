//! Self-describing storage encoding (`SType` / `Encoding` / `ScaleSpec`).
//!
//! `DType` (see [`crate::dtype`]) is the LOGICAL element type — "what is a
//! value". `SType` is orthogonal: it describes HOW those logical elements are
//! physically encoded (block-quantized, sub-byte-packed, …). An empty `SType`
//! means "plain": the bytes are a dense array of `DType`, no extra interpretation.
//!
//! Design: `docs/session-prompts/self-describing-storage-plan.md` (LOCKED
//! DECISION 2026-06-18). The SCHEME is self-describing on the tensor; the scale
//! VALUES are a sibling graph operand (model B); FDX re-unites them at the kernel
//! boundary (`SType::to_fdx`, behind the `dlpack` feature — step 3).

use smallvec::SmallVec;

use crate::dtype::DType;
use crate::quant_scale::ScaleGranularity;
use crate::quantized::GgmlDType;

/// A REQUIREMENT for a sibling per-block / per-axis scale operand — NOT a
/// pointer to one. Says "I need an absmax operand of this dtype + granularity";
/// the consuming op binds the actual operand, and FDX fills the concrete
/// `scale_buffer` index at projection (step 4). The per-block scale SHAPE is
/// DERIVED from the base shape + the layer's `block_shape`, not stored here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScaleSpec {
    /// Element dtype of the required scale operand (commonly `F32`).
    pub dtype: DType,
    /// Granularity of the required scale operand. For `AffineBlock` the block
    /// grain rides the layer's `block_shape`; this is the coarse dispatch-key
    /// granularity (FDX keeps `PerBlock` MX-only — see spec §6.2).
    pub granularity: ScaleGranularity,
}

/// ONE encoding layer. Holds ONLY static descriptors (geometry, scheme, dtype
/// codes, scale REQUIREMENTS) — NEVER bulk data and NEVER scale VALUES. Small,
/// `Eq + Hash` so it can feed structure keys / plan caches.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// GGUF / ggml block format. Scale is baked INLINE in each block struct;
    /// one self-contained buffer, no separate scale operand. Maps to FDX
    /// `GGML_BLOCK` (family 0). `ggml_dtype` IS the format.
    GgmlBlock { ggml_dtype: GgmlDType },

    /// NF4 / QLoRA-style block-grained affine. Maps to FDX `AFFINE_BLOCK`
    /// (family 4): low-bit packed data + a SEPARATE per-block absmax scale
    /// operand (model B). `packed` is the sub-byte storage code (`DType::F4`
    /// for 4-bit; FDX has no distinct NF4 code in v1 — see plan deferred list).
    AffineBlock {
        /// Sub-byte packed storage code (e.g. `DType::F4`).
        packed: DType,
        /// Block extent along each quantized axis (QLoRA default `[64]`).
        block_shape: SmallVec<[u32; 2]>,
        /// The REQUIREMENT for the sibling per-block absmax operand.
        scale: ScaleSpec,
        /// Asymmetric affine zero-point requirement; `None` for symmetric
        /// (NF4 is symmetric → `None`).
        zero_point: Option<ScaleSpec>,
    },

    /// RESERVED placeholder for MX block-scaled (FDX `MX`, family 1). Declared
    /// for shape only; NOT wired in v1 (see plan deferred list).
    Mx,
    // Reserved for LATER (do NOT implement now): AffineInt, AffineFloat, Compressed.
}

impl Encoding {
    /// Whether this layer needs a sibling scale operand bound by the consuming
    /// op (true for `AffineBlock`; false for inline GGML).
    pub fn requires_scale_sibling(&self) -> bool {
        matches!(self, Encoding::AffineBlock { .. })
    }
}

/// An ordered stack of [`Encoding`] layers describing how a `Storage`'s bytes
/// are physically encoded. EMPTY = plain (dense `DType`, no extra interpretation)
/// — the default, byte-identical to pre-SType behaviour.
///
/// Named newtype (not a bare field) because it owns: the layer-ordering
/// invariant, the [`SType::to_fdx`] projection (step 3, `dlpack` feature),
/// construction invariants, and room for representation evolution.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct SType(pub SmallVec<[Encoding; 1]>);

impl SType {
    /// The plain (empty) SType — dense `DType`, no extra interpretation.
    pub fn plain() -> Self {
        SType(SmallVec::new())
    }

    /// A single-layer SType.
    pub fn from_layer(e: Encoding) -> Self {
        let mut v = SmallVec::new();
        v.push(e);
        SType(v)
    }

    /// True iff there are no encoding layers (plain dense `DType`).
    pub fn is_plain(&self) -> bool {
        self.0.is_empty()
    }

    /// The layer stack (outermost-first ordering invariant TBD as layers grow;
    /// v1 has at most one layer).
    pub fn layers(&self) -> &[Encoding] {
        &self.0
    }

    /// Whether ANY layer needs a sibling scale operand.
    pub fn requires_scale_sibling(&self) -> bool {
        self.0.iter().any(Encoding::requires_scale_sibling)
    }
}

#[cfg(feature = "dlpack")]
impl SType {
    /// Project this encoding scheme into an `FDXQuant` for the kernel boundary
    /// (the `view()` quant sidecar). The scale BUFFER reference (`scale_buffer`)
    /// is op-context: pass `scale_buffer = Some(idx)` once the consuming op has
    /// bound the sibling scale operand into the FDX buffer table; `None` ⇒
    /// `FDX_BUFFER_NONE` ("scheme described, scale buffer not yet bound" — the
    /// state a bare `view()` emits; binding happens in the consuming-op wiring).
    ///
    /// Returns `None` for a plain SType (no quant sidecar) and for the `Mx`
    /// placeholder (reserved, not wired in v1). v1 reads `self.0.first()` (at
    /// most one layer); stacked-layer projection is future work.
    pub fn to_fdx(&self, scale_buffer: Option<u32>) -> Option<crate::dlpack::sidecar::FDXQuant> {
        use crate::dlpack::codes::*;
        use crate::dlpack::convert::{dtype_to_fdx, ggml_to_fdx, scale_granularity_to_fdx};
        use crate::dlpack::sidecar::FDXQuant;

        let layer = self.0.first()?;
        match layer {
            // GGML: scale baked INLINE; no separate operand, no granularity.
            Encoding::GgmlBlock { ggml_dtype } => Some(FDXQuant {
                family: FDX_QUANT_GGML_BLOCK,
                ggml_dtype: ggml_to_fdx(*ggml_dtype),
                block_ndim: 0,
                _pad0: [0; 3],
                block_shape: [0; 4],
                block_axes: [-1; 4],
                pack_order: 0,
                _pad1: [0; 3],
                scale_present: 0,
                scale_dtype: FDX_DTYPE_NONE,
                scale_placement: FDX_SCALE_PLACEMENT_INLINE,
                scale_granularity: 0,
                _pad2: [0; 3],
                scale_buffer: FDX_BUFFER_INLINE,
                zp_present: 0,
                zp_dtype: FDX_DTYPE_NONE,
                _pad3: 0,
                zp_buffer: FDX_BUFFER_NONE,
                scale_pair_act: 0,
                scale_pair_weight: 0,
                role: 0,
                _pad4: 0,
                reserved: [0; 6],
            }),
            // AFFINE_BLOCK (NF4/QLoRA): low-bit packed data + a SEPARATE per-block
            // absmax scale operand (model B). `scale_buffer` is bound by the op.
            Encoding::AffineBlock { packed: _, block_shape, scale, zero_point } => {
                let mut bshape = [0u32; 4];
                let mut baxes = [-1i32; 4];
                let n = block_shape.len().min(4);
                for i in 0..n {
                    bshape[i] = block_shape[i];
                    baxes[i] = i as i32; // v1: blocks tile leading axes; refine when wiring the op.
                }
                let (zp_present, zp_dtype) = match zero_point {
                    Some(zp) => (1u8, dtype_to_fdx(zp.dtype)),
                    None => (0u8, FDX_DTYPE_NONE),
                };
                Some(FDXQuant {
                    family: FDX_QUANT_AFFINE_BLOCK,
                    ggml_dtype: FDX_DTYPE_NONE, // not GGML
                    block_ndim: n as u8,
                    _pad0: [0; 3],
                    block_shape: bshape,
                    block_axes: baxes,
                    pack_order: 0,
                    _pad1: [0; 3],
                    scale_present: 1,
                    scale_dtype: dtype_to_fdx(scale.dtype),
                    scale_placement: FDX_SCALE_PLACEMENT_SEPARATE_BUFFER, // never INLINE
                    // AFFINE_BLOCK grain rides block_shape, NOT a granularity byte
                    // (spec §6.2); the coarse dispatch-key granularity is kept but
                    // is never PerBlock.
                    scale_granularity: scale_granularity_to_fdx(scale.granularity),
                    _pad2: [0; 3],
                    scale_buffer: scale_buffer.unwrap_or(FDX_BUFFER_NONE),
                    zp_present,
                    zp_dtype,
                    _pad3: 0,
                    zp_buffer: FDX_BUFFER_NONE,
                    scale_pair_act: 0, // stored weight format, not a dynamic matmul pairing
                    scale_pair_weight: 0,
                    role: 0,
                    _pad4: 0,
                    reserved: [0; 6],
                })
            }
            // RESERVED placeholder — not wired in v1.
            Encoding::Mx => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;
    use crate::quantized::GgmlDType;
    use crate::quant_scale::ScaleGranularity;

    /// Default SType is empty = plain (the byte-identical default).
    #[test]
    fn default_stype_is_empty_plain() {
        let s = SType::default();
        assert!(s.is_plain(), "default SType must be plain (empty layer stack)");
        assert_eq!(s.layers().len(), 0);
    }

    /// SType is Eq + Hash so it can feed structure keys / plan caches.
    #[test]
    fn stype_is_eq_and_hash() {
        use std::collections::HashSet;
        let a = SType::from_layer(Encoding::GgmlBlock { ggml_dtype: GgmlDType::Q4_0 });
        let b = SType::from_layer(Encoding::GgmlBlock { ggml_dtype: GgmlDType::Q4_0 });
        assert_eq!(a, b);
        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b), "equal STypes must hash equal");
    }

    /// AffineBlock carries the static descriptor only (geometry + scale REQUIREMENT),
    /// never the scale values. ScaleSpec is a requirement, not a pointer.
    #[test]
    fn affine_block_holds_static_scale_requirement() {
        let enc = Encoding::AffineBlock {
            packed: DType::F4,
            block_shape: smallvec::smallvec![64],
            scale: ScaleSpec { dtype: DType::F32, granularity: ScaleGranularity::PerChannel },
            zero_point: None,
        };
        let s = SType::from_layer(enc.clone());
        assert!(!s.is_plain());
        assert_eq!(s.layers()[0], enc);
        // The packed sub-byte storage is F4; the LOGICAL dtype is NOT here — it
        // lives on Storage.dtype (step 2). Encoding never names the logical float.
        match &s.layers()[0] {
            Encoding::AffineBlock { packed, block_shape, .. } => {
                assert_eq!(*packed, DType::F4);
                assert_eq!(block_shape.as_slice(), &[64u32]);
            }
            _ => panic!("expected AffineBlock"),
        }
    }

    /// GgmlBlock is a single self-contained inline layer (no scale sibling).
    #[test]
    fn ggml_block_is_inline_single_layer() {
        let s = SType::from_layer(Encoding::GgmlBlock { ggml_dtype: GgmlDType::Q4K });
        assert_eq!(s.layers().len(), 1);
        assert!(!s.requires_scale_sibling(),
            "GGML scale is baked inline; no sibling operand required");
    }

    /// AffineBlock requires a scale sibling operand (the absmax).
    #[test]
    fn affine_block_requires_scale_sibling() {
        let s = SType::from_layer(Encoding::AffineBlock {
            packed: DType::F4,
            block_shape: smallvec::smallvec![64],
            scale: ScaleSpec { dtype: DType::F32, granularity: ScaleGranularity::PerChannel },
            zero_point: None,
        });
        assert!(s.requires_scale_sibling());
    }
}

#[cfg(all(test, feature = "dlpack"))]
mod to_fdx_tests {
    use super::*;
    use crate::DType;
    use crate::quantized::GgmlDType;
    use crate::quant_scale::ScaleGranularity;
    use crate::dlpack::codes::*;

    /// A plain SType has no quant projection.
    #[test]
    fn plain_projects_to_none() {
        assert!(SType::plain().to_fdx(None).is_none());
    }

    /// GGML projects to the inline GGML_BLOCK family: scale baked inline, no
    /// separate operand, ggml_dtype carried through.
    #[test]
    fn ggml_projects_inline() {
        let q = SType::from_layer(Encoding::GgmlBlock { ggml_dtype: GgmlDType::Q4_0 })
            .to_fdx(None)
            .expect("GGML projects");
        assert_eq!(q.family, FDX_QUANT_GGML_BLOCK);
        assert_eq!(q.ggml_dtype, FDX_GGML_Q4_0);
        assert_eq!(q.scale_present, 0);
        assert_eq!(q.scale_placement, FDX_SCALE_PLACEMENT_INLINE);
        assert_eq!(q.scale_buffer, FDX_BUFFER_INLINE);
    }

    /// AFFINE_BLOCK projects to the separate-buffer family; `scale_buffer`
    /// reflects the op-bound index (`None` ⇒ FDX_BUFFER_NONE placeholder).
    #[test]
    fn affine_block_projects_separate_buffer() {
        let enc = Encoding::AffineBlock {
            packed: DType::F4,
            block_shape: smallvec::smallvec![64],
            scale: ScaleSpec { dtype: DType::F32, granularity: ScaleGranularity::PerChannel },
            zero_point: None,
        };
        // Unbound (bare view()).
        let q = SType::from_layer(enc.clone()).to_fdx(None).expect("projects");
        assert_eq!(q.family, FDX_QUANT_AFFINE_BLOCK);
        assert_eq!(q.ggml_dtype, FDX_DTYPE_NONE);
        assert_eq!(q.block_ndim, 1);
        assert_eq!(q.block_shape[0], 64);
        assert_eq!(q.block_axes[0], 0);
        assert_eq!(q.scale_present, 1);
        assert_eq!(q.scale_placement, FDX_SCALE_PLACEMENT_SEPARATE_BUFFER);
        assert_eq!(q.scale_dtype, FDX_DTYPE_F32);
        assert_eq!(q.scale_granularity, FDX_SCALE_GRAN_PER_CHANNEL);
        assert_eq!(q.scale_buffer, FDX_BUFFER_NONE);
        assert_eq!(q.zp_present, 0);
        // Bound (op-context supplies the buffer-table index).
        let bound = SType::from_layer(enc).to_fdx(Some(3)).expect("projects");
        assert_eq!(bound.scale_buffer, 3);
    }

    /// Asymmetric affine carries a zero-point requirement.
    #[test]
    fn affine_block_zero_point_projects() {
        let q = SType::from_layer(Encoding::AffineBlock {
            packed: DType::F4,
            block_shape: smallvec::smallvec![32],
            scale: ScaleSpec { dtype: DType::F32, granularity: ScaleGranularity::PerToken },
            zero_point: Some(ScaleSpec { dtype: DType::F32, granularity: ScaleGranularity::PerToken }),
        })
        .to_fdx(None)
        .expect("projects");
        assert_eq!(q.zp_present, 1);
        assert_eq!(q.zp_dtype, FDX_DTYPE_F32);
    }

    /// The reserved Mx placeholder is not wired in v1.
    #[test]
    fn mx_projects_to_none() {
        assert!(SType::from_layer(Encoding::Mx).to_fdx(None).is_none());
    }
}
