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
