// SPDX-License-Identifier: MIT OR Apache-2.0
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
    ArchSku, ElementKind, MAX_RANK, OpCategory, OperandDesc, structure_key_token,
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
        // Baracuda ships the matching kind with its own FP8 tensor-core operand
        // tag (`.e5m2.e5m2.f32`), and `dtype_token` already emits `f8e5m2`, so a
        // decline here would suppress structure-key derivation for a whole FP8
        // family that IS supported. GAP-097 residual: this arm was MISSING
        // entirely and the omission was a hard E0004 that nothing in this
        // environment compiled — see the note below.
        DType::F8E5M2 => ElementKind::Fp8E5M2,
        // GAP-171. `ElementKind::U32` EXISTS at the locked
        // baracuda-kernel-vocab 0.0.1-alpha.78, so declining it here was a
        // decline whose stated reason ("no faithful Baracuda ElementKind") had
        // EXPIRED — correct when written, silently wrong once the seam gained
        // the variant.
        //
        // Un-declining is a BEHAVIOUR change, not a message change: a decline
        // aborts the WHOLE derivation via `?` at the call site, so a cell with
        // a u32 operand went from emitting NOTHING to emitting a key.
        //
        // And it is not an exotic case. `u32` is the seam's gather/scatter
        // INDEX ctype, so every indexed region carries one — `IndexSelect`,
        // scatter/gather, `PagedAttn`'s `block_table` and `context_lens`. The
        // expired decline was silencing that whole class of telemetry.
        DType::U32 => ElementKind::U32,
        // GAP-168(c). `ElementKind::Bool` EXISTS at the locked
        // baracuda-kernel-vocab 0.0.1-alpha.78 — "1-byte storage, 0/non-zero
        // truthiness", which is exactly Fuel's `DType::Bool`. So this is a real
        // mapping, not a decline: declining would have been an EXPIRED decline
        // from birth, the GAP-171 shape with the clock already run out.
        //
        // ⚠️ THIS ARM WAS MISSING and it was a latent E0004, not a silent
        // fallthrough — the match is wildcard-free. It survived the Bool cut
        // because this file compiles only under `telemetry` AND `cuda`, and that
        // combination is built by no gate the cut ran. Same structural hole as
        // GAP-097's missing `F8E5M2` arm, recurring one dtype later.
        //
        // Un-declining is a BEHAVIOUR change: a `None` aborts the whole
        // derivation via `?`, so a cell with a Bool operand went from emitting
        // NOTHING to emitting a key. Comparison outputs are Bool now, so that is
        // every mask-producing cell.
        DType::Bool => ElementKind::Bool,
        // No faithful Baracuda ElementKind — no signal beats a wrong one.
        //
        // VERIFIED against alpha.78's 18 variants rather than assumed: there is
        // no `I16`, no `F8E8M0`, no `F8E6M2`. These declines are still CORRECT,
        // so they must NOT be batched with `U32`'s — the arm shared one comment
        // for two different situations, which is how the expired one hid.
        DType::I16 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 | DType::F8E6M2 => {
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
///
/// ⚠️ **`"90"` and `"90a"` both map to [`ArchSku::Sm90a`] — a non-injective arm
/// on an axis where injectivity is required**, since a capability token NAMES a
/// target and is therefore an identity, not a classification. It is
/// **lossy-but-FORCED rather than a mistake**: there is no `ArchSku::Sm90` at
/// the locked `baracuda-kernel-vocab 0.0.1-alpha.78` to map `"90"` onto. See
/// [`arch_sku_digits`] for the compile-time trap that fires when that changes.
fn map_arch_sku(arch: &str) -> Option<ArchSku> {
    let digits = arch
        .strip_prefix("sm_")
        .or_else(|| arch.strip_prefix("sm"))?;
    Some(match digits {
        "80" => ArchSku::Sm80,
        "89" => ArchSku::Sm89,
        // NOT splittable today — see `arch_sku_digits`. `"90"` is the ALIAS;
        // `"90a"` is this SKU's canonical spelling.
        "90" | "90a" => ArchSku::Sm90a,
        _ => return None,
    })
}

/// The canonical arch-tag digits of each [`ArchSku`] **this build** knows.
///
/// # This function exists in order to FAIL TO COMPILE
///
/// [`map_arch_sku`] matches on Fuel's arch-tag **string**, so it is
/// *structurally* incapable of noticing a new `ArchSku`: its scrutinee is a
/// `&str`, and a variant added upstream changes nothing it can see. This
/// match's scrutinee **is** `ArchSku`, so it is the one place in Fuel where a
/// variant added to a vocabulary we do not own becomes an `E0004` instead of a
/// silent change in behaviour.
///
/// Upstream intends precisely this: `baracuda-kernel-vocab`'s own doc calls
/// `ArchSku` *"intentionally NOT `#[non_exhaustive]`"*, because a new arch
/// *"deserves to surface as a build break across every match site"* (Blackwell
/// `Sm100a` is on their roadmap). Verified at the locked `0.0.1-alpha.78`,
/// which carries exactly `Sm80` / `Sm89` / `Sm90a`.
///
/// # What to do when this breaks
///
/// It breaks on a **baracuda bump**, not on a Fuel change — that is the point
/// (GAP-179). When it does:
///
/// 1. add the arm here;
/// 2. add the arm to [`map_arch_sku`];
/// 3. **if the new variant is `Sm90`, SPLIT the `"90" | "90a"` arm.** Today
///    both spellings collapse onto `Sm90a` because there is no `Sm90` to map
///    `"90"` onto. The moment there is, the collapse stops being merely lossy
///    and becomes a **wrong answer**: `sm_90` would claim Hopper-*specialized*
///    kernels that a portable-baseline `sm90` target was never built for.
///    `Sm90` already exists in `unpopped-vocab 0.2.0`, so the chain is
///    unpopped-vocab → baracuda re-publishes its vocab → Fuel bumps.
///
/// Nothing inside Fuel changes to cause that, which is why it needs a trap
/// rather than vigilance: without this, the first symptom would be a mis-keyed
/// structure key, not a build error.
// Deliberately unused in production — being type-checked IS the job. Dead code
// is still exhaustiveness-checked, so the `E0004` fires on a plain
// `cargo check`, not only when tests are compiled.
#[allow(dead_code)]
fn arch_sku_digits(sku: ArchSku) -> &'static str {
    match sku {
        ArchSku::Sm80 => "80",
        ArchSku::Sm89 => "89",
        // `Sm90a`, NOT `Sm90` — see the split note above.
        ArchSku::Sm90a => "90a",
    }
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
        "SoftmaxLastDim"
        | "SoftmaxLastDimBackward"
        | "LogSoftmaxLastDim"
        | "LogSoftmaxLastDimBackward" => OpCategory::Softmax,
        "RmsNormLastDim"
        | "RmsNormLastDimBackward"
        | "LayerNormLastDim"
        | "LayerNormLastDimBackward" => OpCategory::Normalization,
        "SumReduce"
        | "MaxReduce"
        | "MinReduce"
        | "MeanReduce"
        | "ReduceSumTo"
        | "ReduceMaxTo"
        | "ReduceMaxToBackward"
        | "ArgMaxDim"
        | "ArgMinDim" => OpCategory::Reduction,
        "CumSum" | "SelectiveScan" | "SsdChunkScan" => OpCategory::Scan,
        "IndexSelect" | "Gather" | "IndexAdd" | "ScatterAdd" | "MaskedFill" => OpCategory::Indexing,
        "Flip" | "Roll" | "Pad" | "PadBackward" | "Triu" | "Tril" | "Concat" | "Copy"
        | "WriteSlice" | "WriteSliceRotating" | "WriteSliceDoff" => OpCategory::ShapeLayout,
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
        assert!(
            mapped.symbolic.is_none(),
            "v1 does not fabricate symbolic facts"
        );
    }

    /// **GAP-171, asserted as BEHAVIOUR rather than as a table entry — because
    /// the behaviour is what changes.**
    ///
    /// A decline in `map_element_kind` aborts the WHOLE derivation (`?` in
    /// `map_operand`, then `?` in `structure_key`), so un-declining `U32` does
    /// not add a field to an existing key: it turns a cell that emitted
    /// **nothing** into one that emits a key. Telemetry rows that never existed
    /// start existing. A test that only checked the mapping table would not
    /// have said that.
    #[test]
    fn a_u32_operand_now_derives_a_key_where_it_previously_derived_none() {
        let p = BaracudaStructureKeyProvider;
        let u32s = [
            contig_of(&[128, 256], DType::U32),
            contig_of(&[128, 256], DType::U32),
        ];
        let t = p
            .structure_key("MatMul", &u32s, "sm_89")
            .expect("a u32 operand must derive a key — before GAP-171 this was None");
        assert!(!t.0.is_empty(), "token must be non-empty");

        // The dtype must actually REACH the key. Without this, a mapping that
        // silently collapsed u32 onto some other kind would still pass above.
        let f16s = [
            contig_of(&[128, 256], DType::F16),
            contig_of(&[128, 256], DType::F16),
        ];
        assert_ne!(
            t,
            p.structure_key("MatMul", &f16s, "sm_89")
                .expect("f16 derives"),
            "u32 must key DIFFERENTLY from f16, or the dtype is being dropped",
        );

        // CONTROL — the rest of that decline arm is untouched, so nobody
        // batches it. `I16`/`F8E8M0`/`F8E6M2` have no `ElementKind` at the
        // locked baracuda-kernel-vocab 0.0.1-alpha.78 (verified against the
        // enum's 18 variants), so their declines are still CORRECT, not stale.
        for dt in [DType::I16, DType::F8E8M0, DType::F8E6M2] {
            assert_eq!(
                map_element_kind(dt),
                None,
                "{dt:?} has no faithful ElementKind at alpha.78 and must stay declined",
            );
        }
    }

    /// The dtype mapping table (representative + a decline).
    #[test]
    fn element_kind_mapping() {
        assert_eq!(map_element_kind(DType::F16), Some(ElementKind::F16));
        assert_eq!(map_element_kind(DType::BF16), Some(ElementKind::Bf16));
        assert_eq!(map_element_kind(DType::I8), Some(ElementKind::S8));
        assert_eq!(map_element_kind(DType::F8E4M3), Some(ElementKind::Fp8E4M3));
        // GAP-097 residual: this arm was missing entirely, which was a hard
        // E0004 that only `--features telemetry,cuda` could ever surface.
        // Asserted as a MAPPING, not a decline — Baracuda ships the kind, so
        // declining would have suppressed a supported FP8 family.
        assert_eq!(map_element_kind(DType::F8E5M2), Some(ElementKind::Fp8E5M2));
        // GAP-171: this asserted `None` while that was merely what the code
        // DID. It now asserts what is RIGHT — `ElementKind::U32` exists at the
        // locked alpha.78, so declining it was an expired decline.
        assert_eq!(map_element_kind(DType::U32), Some(ElementKind::U32));
        // Still no faithful equivalent ⇒ still declines. Correct, not stale.
        assert_eq!(map_element_kind(DType::F4), None);
    }

    /// Every [`ArchSku`] this build knows is reachable from its own canonical
    /// digits, in both accepted spellings.
    ///
    /// This is what keeps [`arch_sku_digits`] from being satisfiable by an arm
    /// that returns a spelling [`map_arch_sku`] does not accept — the witness
    /// forces a decision, and this forces that decision to be a *correct* one.
    ///
    /// ⚠️ **`KNOWN` is CONVENIENCE, NOT THE ANCHOR.** A hand-written list
    /// cannot detect a variant added upstream — it just stays short, silently.
    /// `arch_sku_digits` can, and breaks *first*, because it is the only
    /// construct here whose scrutinee is the enum itself. If you are reading
    /// this because the list looks stale, the error you should already have
    /// seen is an `E0004` in that match.
    #[test]
    fn every_known_arch_sku_round_trips_through_its_canonical_digits() {
        const KNOWN: &[ArchSku] = &[ArchSku::Sm80, ArchSku::Sm89, ArchSku::Sm90a];

        for &sku in KNOWN {
            let digits = arch_sku_digits(sku);
            assert_eq!(
                map_arch_sku(&format!("sm_{digits}")),
                Some(sku),
                "sm_{digits} must select {sku:?}"
            );
            assert_eq!(
                map_arch_sku(&format!("sm{digits}")),
                Some(sku),
                "sm{digits} (bare form) must select {sku:?}"
            );
        }

        // The CANONICAL direction must be injective, or the round-trip above is
        // satisfiable by a table that collapses two variants onto one spelling
        // — which is exactly the defect this row is about, one level up.
        let mut seen = std::collections::BTreeSet::new();
        for &sku in KNOWN {
            assert!(
                seen.insert(arch_sku_digits(sku)),
                "two ArchSku variants share canonical digits {:?}",
                arch_sku_digits(sku)
            );
        }
        assert_eq!(seen.len(), 3, "non-vacuity: the known-SKU set is not empty");
    }

    // ⚠️ THIS TEST AND `the_bare_90_alias_...` ARE NOT REDUNDANT, and the
    // reason is not visible by reading them. MEASURED by sabotage: setting
    // `arch_sku_digits(Sm90a)` to the ALIAS `"90"` leaves the round-trip above
    // GREEN — because `map_arch_sku` accepts `"90"` too, so a wrong canonical
    // spelling still round-trips. Only the explicit
    // `arch_sku_digits(Sm90a) == "90a"` assertion catches it. Do not delete
    // that one on the grounds that this one covers it; it does not.

    /// The one NON-INJECTIVE arm, pinned together with its reason and expiry.
    ///
    /// `"90"` and `"90a"` are two distinct targets sharing one internal value.
    /// By Fuel's own cut — injectivity is *mandatory* where the output is an
    /// IDENTITY, optional where it is a CLASSIFICATION — a capability token
    /// names a target, so injectivity is required and this arm violates it.
    ///
    /// It is nevertheless **correct today**, because there is nothing else to
    /// map `"90"` onto. That is the distinction this test exists to record:
    /// LATENT, not live-wrong. [`arch_sku_digits`] is what converts it from a
    /// fact someone has to remember into a build failure.
    #[test]
    fn the_bare_90_alias_is_a_forced_collapse_not_a_mapping() {
        assert_eq!(
            map_arch_sku("sm_90"),
            map_arch_sku("sm_90a"),
            "the collapse is the documented state; if this now FAILS, the arm \
             was split and this test should be deleted, not repaired"
        );
        assert_eq!(map_arch_sku("sm_90"), Some(ArchSku::Sm90a));
        // `"90a"` is the CANONICAL spelling of that SKU, so `"90"` is the extra
        // one. That asymmetry is what makes the eventual split mechanical:
        // `Sm90` takes `"90"` and this arm keeps `"90a"`.
        assert_eq!(arch_sku_digits(ArchSku::Sm90a), "90a");
    }

    /// The arch + op-class mapping (incl. the underscore-tolerant form and the
    /// arity-driven elementwise families).
    #[test]
    fn arch_and_op_class_mapping() {
        assert_eq!(map_arch_sku("sm_80"), Some(ArchSku::Sm80));
        assert_eq!(map_arch_sku("sm_89"), Some(ArchSku::Sm89));
        // ⚠️ This asserts what the code DOES, not what is RIGHT — the same
        // shape as the `map_element_kind(U32)` line above before GAP-171 fixed
        // it. `sm_90` landing on the Hopper-SPECIALIZED SKU is a forced
        // collapse, not a mapping; see `the_bare_90_alias_is_a_forced_collapse`
        // for the reason and the expiry. Kept because it pins today's
        // behaviour, and it MUST change when `ArchSku::Sm90` lands (GAP-179).
        assert_eq!(map_arch_sku("sm_90"), Some(ArchSku::Sm90a));
        assert_eq!(map_arch_sku("sm89"), Some(ArchSku::Sm89));
        assert_eq!(map_arch_sku("cpu"), None);
        assert!(matches!(
            map_op_category("MatMul", 2),
            Some(OpCategory::Gemm)
        ));
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
        // GAP-171: this used `U32` as the unmappable example, and `U32` now
        // MAPS (`ElementKind::U32` exists at alpha.78). The assertion's intent
        // is unchanged — an unmappable dtype must still yield no key — so it
        // moves to a dtype that genuinely is one rather than being deleted.
        //
        // `I16` is verified unmappable against alpha.78's 18 `ElementKind`
        // variants: the seam has S8/U8/I32/I64/U32 and no 16-bit int.
        //
        // That this test failed at all is the point: un-declining `U32` was a
        // BEHAVIOUR change, and it reached a test that never mentioned GAP-171.
        let i16op = [contig_of(&[8, 16], DType::I16)];
        assert!(
            p.structure_key("ReluElementwise", &i16op, "sm_89")
                .is_none(),
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
            p.structure_key("ReluElementwise", &[big], "sm_89")
                .is_none(),
            "rank > MAX_RANK must decline, not panic"
        );
    }
}
