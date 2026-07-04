//! The Baracuda-backed [`StructureKeyProvider`] — the live wire into Baracuda's
//! shipped `structure_key` keying function.
//!
//! Fuel projects each live `(Layout, DType)` into an [`FdxOperandDesc`]
//! (`structure_key.rs`); this provider maps that RAW descriptor onto Baracuda's
//! `baracuda_kernels_types::OperandDesc` field-by-field and CALLS Baracuda's
//! single canonical `structure_key_token(op, operands, arch)`. Fuel NEVER
//! derives or parses the token (K1 opacity): the string comes back from Baracuda
//! and is wrapped verbatim in a [`StructureKeyToken`]. Because both Fuel's
//! telemetry tag and Baracuda's build matrix call the SAME keyer, they join on
//! the same token by construction.
//!
//! `structure_key` is **pure host code** — a keying function over operand
//! descriptors in the `baracuda-kernels-types` types crate (no FFI, no device),
//! so this provider needs no GPU. It is gated on `feature = "cuda"` only because
//! its output is meaningful solely for a CUDA target arch (Baracuda's build
//! matrix is CUDA kernels); a CPU-only build keeps the
//! [`NullStructureKeyProvider`](super::structure_key::NullStructureKeyProvider).
//!
//! # Honest-`None` posture (no signal beats a wrong signal)
//!
//! The provider returns `None` (no token ⇒ no demand signal) whenever it cannot
//! form a FAITHFUL key, rather than fabricate one. It NEVER panics. The declining
//! cases:
//! - an `op_class` with no Baracuda [`OpCategory`] (an unmapped op family);
//! - an `arch` outside Baracuda's shipped SKUs (`sm_80` / `sm_89` / `sm_90`; a
//!   CPU realize tags `"cpu"`, which has no build matrix);
//! - an operand `dtype` with no Baracuda [`ElementKind`] (`u32` / `i16` / the
//!   MX6 / MX4 / E8M0 formats have no equivalent);
//! - an operand rank above Baracuda's [`MAX_RANK`] (8), or a malformed
//!   shape/stride pair (which would otherwise panic `OperandDesc::new`).
//!
//! # FdxOperandDesc → OperandDesc mapping (field-by-field)
//!
//! | `FdxOperandDesc`   | `OperandDesc`      | note                              |
//! |--------------------|--------------------|-----------------------------------|
//! | `shape`            | `shape[..rank]`    | raw extents, `i64`                |
//! | `strides`          | `strides[..rank]`  | signed (0 bcast, < 0 flip)        |
//! | `dtype`            | `dtype`            | via [`map_element_kind`]          |
//! | `align_bytes`      | `align_bytes`      | Fuel's alignment estimate         |
//! | (shape.len())      | `rank`             | ≤ [`MAX_RANK`], else decline      |
//! | —                  | `quant`            | `None` (v1: key ignores quant)    |
//! | —                  | `symbolic`         | `None` (v1: key ignores symbolic) |
//!
//! The derived `contiguity` / `broadcast` / `flipped` booleans on
//! `FdxOperandDesc` are DELIBERATELY not read here — Baracuda re-derives those
//! (and the richer vec-width / divisibility axes) from the raw `shape`/`strides`,
//! so Fuel never double-derives the key (K1).
//!
//! # Known fidelity gaps (documented, not fabricated)
//!
//! - **Operand set** = the call site's operands (the node's INPUTS). Baracuda's
//!   `structure_key` treats the slice as "inputs then output"; the emission site
//!   passes inputs only, so the key is over the input structure. Deterministic
//!   and discriminating, but the output operand's structure is not yet folded in.
//! - **`align_bytes`** is an estimate (lazy DAG has no live pointer) — see
//!   [`super::structure_key::estimate_align_bytes`]. Affects only the key's
//!   vec-width axis.
//! - **arch tag** — Fuel emits `sm_<cc>` (e.g. `"sm_89"`); Baracuda's SKU token
//!   is `"sm89"`. [`map_arch_sku`] accepts either form and maps to the single
//!   SKU Baracuda ships per family (`sm_90` → `Sm90a`).

use baracuda_kernels_types::{
    structure_key_token, ArchSku, ElementKind, OpCategory, OperandDesc, MAX_RANK,
};
use fuel_ir::DType;

use super::structure_key::{FdxOperandDesc, StructureKeyProvider, StructureKeyToken};

/// The live provider that calls Baracuda's canonical `structure_key`. Stateless.
#[derive(Debug, Clone, Copy, Default)]
pub struct BaracudaStructureKeyProvider;

impl StructureKeyProvider for BaracudaStructureKeyProvider {
    fn structure_key(
        &self,
        op_class: &str,
        operands: &[FdxOperandDesc],
        arch: &str,
    ) -> Option<StructureKeyToken> {
        // Map the three inputs into Baracuda's vocabulary. Any axis we cannot map
        // faithfully ⇒ no key (never a fabricated token).
        let op = map_op_category(op_class, operands.len())?;
        let arch = map_arch_sku(arch)?;
        let mut mapped = Vec::with_capacity(operands.len());
        for od in operands {
            mapped.push(map_operand(od)?);
        }
        // Call Baracuda's single canonical keyer and wrap its opaque token as-is.
        Some(StructureKeyToken(structure_key_token(op, &mapped, arch)))
    }
}

/// Map one [`FdxOperandDesc`] onto Baracuda's [`OperandDesc`]. Returns `None` for
/// an unmappable dtype, an over-rank operand, or a malformed shape/stride pair —
/// never panics (Baracuda's `OperandDesc::new` would panic on `rank > MAX_RANK`
/// or a short stride slice).
fn map_operand(od: &FdxOperandDesc) -> Option<OperandDesc> {
    let dtype = map_element_kind(od.dtype)?;
    let rank = od.shape.len();
    if rank > MAX_RANK || od.strides.len() != rank {
        return None;
    }
    Some(OperandDesc::new(
        rank,
        &od.shape,
        &od.strides,
        dtype,
        od.align_bytes,
    ))
}

/// Map a Fuel [`DType`] to a Baracuda [`ElementKind`]. Exhaustive (no `_` arm) so
/// a new Fuel dtype forces a mapping decision here rather than silently keying
/// wrong. Dtypes with no faithful Baracuda equivalent decline (`None`).
fn map_element_kind(dt: DType) -> Option<ElementKind> {
    Some(match dt {
        DType::U8 => ElementKind::U8,
        DType::I8 => ElementKind::S8,
        DType::I32 => ElementKind::I32,
        DType::I64 => ElementKind::I64,
        DType::BF16 => ElementKind::Bf16,
        DType::F16 => ElementKind::F16,
        DType::F32 => ElementKind::F32,
        DType::F64 => ElementKind::F64,
        DType::F8E4M3 => ElementKind::Fp8E4M3,
        // No faithful Baracuda ElementKind — no signal beats a wrong one.
        DType::U32 | DType::I16 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
            return None;
        }
    })
}

/// Map Fuel's arch tag (`hooks::arch_tag`) to a Baracuda [`ArchSku`].
///
/// Fuel emits `sm_<major><minor>` (e.g. `"sm_89"`) or `"cpu"`; Baracuda's SKU
/// tokens are `"sm80"` / `"sm89"` / `"sm90a"`. Accept either the underscore or
/// bare form and map to the single SKU Baracuda ships per family. `"cpu"` and
/// any unshipped SKU decline (no build matrix ⇒ no key).
fn map_arch_sku(arch: &str) -> Option<ArchSku> {
    let digits = arch.strip_prefix("sm_").or_else(|| arch.strip_prefix("sm"))?;
    Some(match digits {
        "80" => ArchSku::Sm80,
        "89" => ArchSku::Sm89,
        "90" | "90a" => ArchSku::Sm90a,
        _ => return None,
    })
}

/// Map a Fuel `OpKind` Debug name (the `op_class` string the emission site
/// stamps via `format!("{op_kind:?}")`) to a Baracuda [`OpCategory`].
///
/// Non-elementwise families are keyed by exact name; the elementwise / in-place
/// families take their arity from the live operand count (`n_inputs`). An
/// unrecognized name declines (`None`) — an honest "no category" that keys no
/// signal rather than a wrong one.
fn map_op_category(op_class: &str, n_inputs: usize) -> Option<OpCategory> {
    let cat = match op_class {
        "MatMul" | "FusedLinear" | "QMatMul" | "Nf4Matmul" => OpCategory::Gemm,
        "Conv2D" | "ConvTranspose2D" | "CausalConv1d" => OpCategory::Convolution,
        "FlashAttn" | "FlashAttnBackwardQ" | "FlashAttnBackwardK" | "FlashAttnBackwardV"
        | "PagedAttn" | "Rope" => OpCategory::Attention,
        "SoftmaxLastDim" | "SoftmaxLastDimBackward" | "LogSoftmaxLastDim"
        | "LogSoftmaxLastDimBackward" => OpCategory::Softmax,
        "RmsNormLastDim" | "RmsNormLastDimBackward" | "LayerNormLastDim"
        | "LayerNormLastDimBackward" => OpCategory::Normalization,
        "SumReduce" | "MaxReduce" | "MinReduce" | "MeanReduce" | "ReduceSumTo" | "ReduceMaxTo"
        | "ReduceMaxToBackward" | "ArgMaxDim" | "ArgMinDim" => OpCategory::Reduction,
        "CumSum" | "SelectiveScan" | "SsdChunkScan" => OpCategory::Scan,
        "IndexSelect" | "Gather" | "IndexAdd" | "ScatterAdd" | "MaskedFill" => OpCategory::Indexing,
        "Flip" | "Roll" | "Pad" | "PadBackward" | "Triu" | "Tril" | "Concat" | "Copy"
        | "WriteSlice" | "WriteSliceRotating" => OpCategory::ShapeLayout,
        "FusedSoftmaxCrossEntropy" => OpCategory::Loss,
        "Where" => OpCategory::TernaryElementwise,
        // Cast / affine are per-element transforms with no dedicated category.
        "Cast" | "Affine" | "InplaceAffine" => OpCategory::UnaryElementwise,
        // Elementwise / in-place families: arity comes from the live operands.
        other if other.ends_with("Elementwise") || other.ends_with("Inplace") => match n_inputs {
            1 => OpCategory::UnaryElementwise,
            2 => OpCategory::BinaryElementwise,
            3 => OpCategory::TernaryElementwise,
            _ => return None,
        },
        _ => return None,
    };
    Some(cat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::{Layout, Shape, StrideVec};

    fn contig_of(dims: &[usize], dt: DType) -> FdxOperandDesc {
        FdxOperandDesc::from_layout(&Layout::contiguous(Shape::from(dims.to_vec())), dt)
    }
    fn contig_f16(dims: &[usize]) -> FdxOperandDesc {
        contig_of(dims, DType::F16)
    }
    fn flipped_f16(dims: &[usize]) -> FdxOperandDesc {
        // Row-major contiguous strides with the inner axis NEGATED — an Op::Flip
        // view (a live reverse-stride demand axis).
        let mut stride: StrideVec = Shape::from(dims.to_vec()).stride_contiguous();
        if let Some(last) = stride.last_mut() {
            *last = -*last;
        }
        let offset = dims.last().copied().unwrap_or(1).saturating_sub(1);
        let layout = Layout::new(Shape::from(dims.to_vec()), stride, offset);
        FdxOperandDesc::from_layout(&layout, DType::F16)
    }

    /// BORN-RED headline: a contiguous f16 (matmul-ish) operand pair yields a
    /// non-empty token that is STABLE across two calls (determinism).
    #[test]
    fn contiguous_f16_pair_yields_stable_nonempty_token() {
        let p = BaracudaStructureKeyProvider;
        let ops = [contig_f16(&[128, 256]), contig_f16(&[128, 256])];
        let t1 = p
            .structure_key("MatMul", &ops, "sm_89")
            .expect("linked provider must yield a token");
        assert!(!t1.0.is_empty(), "token must be non-empty");
        let t2 = p
            .structure_key("MatMul", &ops, "sm_89")
            .expect("token on the second call too");
        assert_eq!(t1, t2, "structure_key is deterministic");
    }

    /// Sanity (not parsing): a structurally different operand set (a flipped
    /// operand) keys to a DIFFERENT token — the flip demand axis flows through.
    #[test]
    fn different_operand_structure_yields_different_token() {
        let p = BaracudaStructureKeyProvider;
        let contig = [contig_f16(&[128, 256]), contig_f16(&[128, 256])];
        let flipped = [flipped_f16(&[128, 256]), contig_f16(&[128, 256])];
        let a = p.structure_key("MatMul", &contig, "sm_89").unwrap();
        let b = p.structure_key("MatMul", &flipped, "sm_89").unwrap();
        assert_ne!(a, b, "a flipped operand must key differently");
    }

    /// Mapping fidelity: `FdxOperandDesc` → Baracuda `OperandDesc` field-by-field.
    #[test]
    fn maps_fdx_operand_desc_to_baracuda_operand_desc_field_for_field() {
        let od = contig_f16(&[8, 16]);
        let mapped = map_operand(&od).expect("mappable");
        assert_eq!(mapped.rank, 2);
        assert_eq!(&mapped.shape[..2], &[8i64, 16]);
        assert_eq!(&mapped.strides[..2], &[16i64, 1]);
        assert_eq!(mapped.dtype, ElementKind::F16);
        assert_eq!(mapped.align_bytes, od.align_bytes);
        assert!(mapped.quant.is_none(), "v1 does not fabricate quant facts");
        assert!(mapped.symbolic.is_none(), "v1 does not fabricate symbolic facts");
    }

    /// The dtype mapping table (representative + a decline).
    #[test]
    fn element_kind_mapping() {
        assert_eq!(map_element_kind(DType::F16), Some(ElementKind::F16));
        assert_eq!(map_element_kind(DType::BF16), Some(ElementKind::Bf16));
        assert_eq!(map_element_kind(DType::I8), Some(ElementKind::S8));
        assert_eq!(map_element_kind(DType::F8E4M3), Some(ElementKind::Fp8E4M3));
        // No faithful equivalent ⇒ decline.
        assert_eq!(map_element_kind(DType::U32), None);
        assert_eq!(map_element_kind(DType::F4), None);
    }

    /// The arch + op-class mapping (incl. the underscore-tolerant form and the
    /// arity-driven elementwise families).
    #[test]
    fn arch_and_op_class_mapping() {
        assert_eq!(map_arch_sku("sm_80"), Some(ArchSku::Sm80));
        assert_eq!(map_arch_sku("sm_89"), Some(ArchSku::Sm89));
        assert_eq!(map_arch_sku("sm_90"), Some(ArchSku::Sm90a));
        assert_eq!(map_arch_sku("sm89"), Some(ArchSku::Sm89));
        assert_eq!(map_arch_sku("cpu"), None);
        assert!(matches!(map_op_category("MatMul", 2), Some(OpCategory::Gemm)));
        assert!(matches!(
            map_op_category("AddElementwise", 2),
            Some(OpCategory::BinaryElementwise)
        ));
        assert!(matches!(
            map_op_category("ReluElementwise", 1),
            Some(OpCategory::UnaryElementwise)
        ));
        assert!(matches!(
            map_op_category("Where", 3),
            Some(OpCategory::TernaryElementwise)
        ));
        assert_eq!(map_op_category("TotallyUnknownOp", 1), None);
    }

    /// Honest `None`: unmapped op family, CPU arch, and unmappable dtype each
    /// key no signal (never a fabricated token).
    #[test]
    fn unmappable_inputs_yield_none() {
        let p = BaracudaStructureKeyProvider;
        let f16 = [contig_f16(&[8, 16])];
        assert!(
            p.structure_key("TotallyUnknownOp", &f16, "sm_89").is_none(),
            "unmapped op family ⇒ no key"
        );
        assert!(
            p.structure_key("ReluElementwise", &f16, "cpu").is_none(),
            "CPU arch has no Baracuda build matrix ⇒ no key"
        );
        let u32op = [contig_of(&[8, 16], DType::U32)];
        assert!(
            p.structure_key("ReluElementwise", &u32op, "sm_89").is_none(),
            "unmappable dtype ⇒ no key"
        );
    }

    /// An over-rank operand declines instead of panicking Baracuda's
    /// `OperandDesc::new` (rank 9 > MAX_RANK 8).
    #[test]
    fn over_rank_operand_declines_never_panics() {
        let p = BaracudaStructureKeyProvider;
        let big = contig_f16(&[2, 2, 2, 2, 2, 2, 2, 2, 2]);
        assert!(
            p.structure_key("ReluElementwise", &[big], "sm_89").is_none(),
            "rank > MAX_RANK must decline, not panic"
        );
    }
}
