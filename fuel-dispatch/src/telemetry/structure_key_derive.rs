// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fuel's INDEPENDENT `structure_key` derivation — the second implementation
//! for the two-implementation freeze-gate (KISS-CLASSIFY §6.6/§6.7).
//!
//! This is deliberately **Baracuda-free**: it recomputes the same `sk4` token
//! from Fuel's own [`FdxOperandDesc`] projection, with **no** `baracuda_kernels_*`
//! import, so a byte-match against Baracuda's emitted token is a genuine
//! two-implementation agreement. (K1 opacity — "Fuel never derives the key" —
//! governs the DISPATCH seam in [`super::structure_key`]; the freeze-gate is the
//! deliberate exception: Fuel derives the key independently *to check* it, never
//! to route.)
//!
//! Schema version: **sk4** (KISS-CLASSIFY §6.1/§6.4, PR #131). Derived from the
//! spec clauses, not from Baracuda's implementation.
//!
//! The **sk4→sk4** delta this deriver implements — a pure respelling of the
//! §6.1 dtype vocabulary plus the version prefix, with the key's field
//! structure unchanged:
//! - every token re-prefixes `sk4|` → `sk4|` (§6.7-0002, canonical spelling);
//! - the signed integers are **i-prefixed**: `s8`→`i8`, `s16`→`i16`
//!   (§6.1-0001; the table annotates each as "sk4 `s8`"/"sk4 `s16`"). Easy to
//!   miss, because attention lands on the FP8 rows and `s16` does not *look*
//!   retired the way a bare `e4m3` does;
//! - the FP8 tokens gain the `f8` width prefix: `e4m3fn` → `f8e4m3fn`
//!   (§3.1.2), and `f8e5m2` joins **unsuffixed** — only `fnuz` deviates from
//!   IEEE E5M2, so E5M2 carries no variant suffix (§3.1.5);
//! - the two 8-bit MX **scales** `f8e8m0` / `f8e6m2` are added to the closed
//!   set (additive at sk4). Fuel has both dtypes, so they are now **emitted**
//!   rather than declined — which means cells whose first operand is one of
//!   them go from producing **no key at all** to producing a well-formed one
//!   (the decline is a `?` on the whole derivation, not a per-field fallback);
//! - the block-scoped sub-byte **element** formats (`F6E2M3`/`F6E3M2`/`F4`)
//!   remain outside the closed set and keep declining (PR-2, GAP-153).
//!
//! Historical (the earlier **sk2→sk4** delta this deriver already carried):
//! - every token re-prefixed `sk2|` → `sk3|`;
//! - the `gem` contraction field grows the precision/compute coordinates:
//!   `c<m><n><k>/<kdiv>[/b<class>]/<wdt>/<acc>/<out>/<mp>` — six `/`-parts
//!   non-batched, seven batched (§6.7-0006). This settles decision D1, so the
//!   deriver's former `gem` decline is replaced by a real derivation;
//! - the FP8 spellings are variant-explicit (§6.1-0001): bare `e4m3` retires
//!   in favor of `e4m3fn`; the AMD `fnuz` variants are **reserved** (their use
//!   typed-declines at this schema version). Fuel's [`DType`] carries no fnuz
//!   variant, so this emitter can never produce one (enforced by test); it has
//!   no token *parse* path, so the reserved-on-parse arm is not applicable here.
//!
//! This rebuild also aligns the derivation with the pinned §6.5 algorithms
//! where the sk2-era code had latent divergences (none reachable by the sk2
//! freeze-gate cell): the innermost axis is axis `rank−1` (§6.3-0011), the
//! divisibility ladder carries the `E ≥ N` guard so `E = 0` buckets `da`
//! (§6.5-0012), a reduction cell's reduced innermost axis derives `v1`
//! (§6.5-0009(b)), the layout tag follows the 4-step `|stride|` algorithm
//! (§6.5-0002), the work class reads the iteration frame (§6.5-0010), and
//! rank-deficient operands are right-aligned into the frame (§6.6-0013).

use super::structure_key::FdxOperandDesc;
use fuel_ir::DType;

/// The reduced-axis set of a `red` cell — the reduce field (§6.6-0009 / §6.7-0005).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceAxes {
    /// Every axis reduced → `rall`.
    All,
    /// Only the trailing (innermost) axis → `rlast`.
    TrailingAxis,
    /// An explicit keepdim bitmask for any other axis set → `x<hh>`.
    Keepdim(u8),
}

/// The math-precision key coordinate of a `gem` cell — `<mp>` in the sk4
/// contraction field (§6.7-0006), resolving to the KISS-Ops §6.17
/// MathPrecision value per `(primary_dtype, target)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemMathPrecision {
    /// `st` — bit-stable: no input rounding (§6.17-0006).
    BitStable,
    /// `rm` — reduced-mantissa-permitted (on an `f32` primary at
    /// `cuda:sm80+`, TF32: 10 retained mantissa bits, RNE; §6.17-0006).
    ReducedMantissa,
}

impl GemMathPrecision {
    fn code(self) -> &'static str {
        match self {
            GemMathPrecision::BitStable => "st",
            GemMathPrecision::ReducedMantissa => "rm",
        }
    }
}

/// The caller-supplied role hints of a dense-contraction (`gem`) cell
/// (§6.6-0012/-0016): the M/N/K axis-role extents (an implementation MUST NOT
/// infer M/N/K from bare operand extents), the conditionally-present batch
/// extent, and the sk4 precision coordinates — weight / accumulator / output
/// dtypes plus the math-precision class (§6.7-0006).
///
/// The dtype coordinates are Fuel [`DType`]s, not spellings: the closed §6.1
/// token set is applied by [`dtype_token`] at emission, so a reserved (`fnuz`)
/// or out-of-set spelling is **unrepresentable** here — build-time closure
/// instead of a parse-time decline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemCell {
    /// M axis extent (caller role hint).
    pub m: i64,
    /// N axis extent (caller role hint).
    pub n: i64,
    /// K (contracted) axis extent (caller role hint).
    pub k: i64,
    /// Batch extent — `Some` iff the cell is batched; a non-batched cell
    /// omits the `b<class>` coordinate entirely (§6.7-0006).
    pub batch: Option<i64>,
    /// Weight dtype coordinate `<wdt>`.
    pub weight_dtype: DType,
    /// Accumulator dtype coordinate `<acc>` (the identity/lookup surface of
    /// the contract's `accumulation_type`, KISS-CONTRACT §6.8-0011).
    pub acc_dtype: DType,
    /// Output dtype coordinate `<out>`.
    pub out_dtype: DType,
    /// Math-precision class `<mp>` ∈ {`st`, `rm`}.
    pub math_precision: GemMathPrecision,
}

/// The **non-contraction** precision coordinate `(acc + mp)` — the sk4
/// optional-trailing field of a non-`gem` cell (KISS-CLASSIFY §6.7-0013,
/// realizing the §6.7-0012 forward requirement).
///
/// Spelled **gem-symmetrically** as `<acc>/<mp>`, and it occupies the *same*
/// optional-trailing slot the contraction group occupies for `gem`. A cell
/// carries **at most one** precision field and the two **never coexist**, so
/// the 9-vs-10 field count resolves on the op-family code alone.
///
/// # Absence means the diagonal, not "unspecified"
///
/// Omitting this is not a gap: per §6.7-0012 an absent coordinate means the
/// accumulator dtype **equals the compute dtype** (the §6.17-0005 diagonal),
/// so every pre-sk4 token keeps its meaning. That is what makes the field
/// additive in the strict sense — no previously-derivable token is affected by
/// its absence — rather than merely appended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccMp {
    /// Accumulator dtype coordinate `<acc>`, from the closed §6.1 set.
    pub acc_dtype: DType,
    /// Math-precision coordinate `<mp>`, the same codes as `gem`'s.
    pub math_precision: GemMathPrecision,
}

impl AccMp {
    /// The default math-precision a non-contraction cell is compared against
    /// for emission purposes.
    ///
    /// **Provenance, deliberately recorded rather than cited:** §6.7-0013 says
    /// the field is emitted when `<mp>` differs from *"the cell's default
    /// math-precision"* and defers to KISS-Ops §6.17-0006 — **which never uses
    /// the word "default"** (positive-controlled: 0 hits in 116 lines). The
    /// value `st` is stated only in a **parenthetical inside a worked example**
    /// in `classify.md` (`<acc>/<mp>` = `f32/st`, "mp at its `st` default").
    /// So this constant is correct-but-underspecified upstream; the
    /// underspecification is raised with KISS. If they pin it normatively this
    /// comment becomes a citation, and if they pin it *differently* this is the
    /// one line that has to change.
    const DEFAULT_MP: GemMathPrecision = GemMathPrecision::BitStable;

    /// §6.7-0013(a): the field is emitted **iff at least one** coordinate
    /// deviates — accumulator ≠ compute dtype, or `<mp>` ≠ default.
    ///
    /// §6.7-0013(d) makes the all-default form **invalid**, so "always emit,
    /// defaults included" is not a safe simplification.
    fn deviates_from(&self, compute_dtype: DType) -> bool {
        self.acc_dtype != compute_dtype || self.math_precision != Self::DEFAULT_MP
    }

    /// §6.7-0013(b): when emitted, **both** slots are spelled explicitly,
    /// including one sitting at its default.
    fn field(&self) -> Option<String> {
        Some(format!(
            "{}/{}",
            dtype_token(self.acc_dtype)?,
            self.math_precision.code(),
        ))
    }
}

/// The op-family a `structure_key` keys on — the KISS-CLASSIFY §6.5-0006
/// 3-letter domain (the subset Fuel can present today). `Reduction` carries its
/// reduce field (§6.6-0009); `Contraction` carries the sk4 [`GemCell`] role
/// hints + precision coordinates (§6.6-0016 / §6.7-0006) — the former
/// pending-D1 decline is settled by the sk4 schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuelOpCategory {
    /// `une` — unary elementwise (GAP-168 op-family axis). Fuel's LARGEST op
    /// family by count: 24 `*Elementwise` unary + 23 `*Inplace` `OpKind`s, plus
    /// `Cast`/`Affine`/`InplaceAffine`, which the production baracuda provider
    /// already folds in here (`baracuda_provider::map_op_category`). This closes
    /// a SPELLING gap on the conformance side; it invents no semantics.
    UnaryElementwise,
    BinaryElementwise,
    TernaryElementwise,
    /// `scn` — scan / prefix ops (GAP-168 op-family axis): `CumSum`,
    /// `SelectiveScan`, `SsdChunkScan`, plus graph-level `Op::Scan`.
    ///
    /// ⚠️ These ops EXIST. The G3 basis gap is that two of them lack a total
    /// `decompose` — a DIFFERENT gap from the family being absent. An earlier
    /// reading took the basis gap to imply Fuel has no scan ops; the published
    /// corpus carries a live `scn` positive and Fuel ships three.
    Scan,
    Reduction(ReduceAxes),
    Contraction(GemCell),
    Normalization,
    Convolution,
    Pooling,
    Indexing,
    ShapeLayout,
    Sorting,
    Fft,
    Linalg,
    Random,
    SegmentOps,
    Softmax,
    Attention,
    Loss,
}

impl FuelOpCategory {
    /// The §6.5-0006 3-letter family code.
    fn code(self) -> &'static str {
        match self {
            FuelOpCategory::UnaryElementwise => "une",
            FuelOpCategory::BinaryElementwise => "bin",
            FuelOpCategory::TernaryElementwise => "ter",
            FuelOpCategory::Scan => "scn",
            FuelOpCategory::Reduction(_) => "red",
            FuelOpCategory::Contraction(_) => "gem",
            FuelOpCategory::Normalization => "nrm",
            FuelOpCategory::Convolution => "cnv",
            FuelOpCategory::Pooling => "pol",
            FuelOpCategory::Indexing => "idx",
            FuelOpCategory::ShapeLayout => "shp",
            FuelOpCategory::Sorting => "srt",
            FuelOpCategory::Fft => "fft",
            FuelOpCategory::Linalg => "lin",
            FuelOpCategory::Random => "rnd",
            FuelOpCategory::SegmentOps => "seg",
            FuelOpCategory::Softmax => "sft",
            FuelOpCategory::Attention => "att",
            FuelOpCategory::Loss => "los",
        }
    }

    /// The reduce field (§6.6-0009): a non-`-` value only for a `red` cell —
    /// every other family emits `-` by construction (§6.6-0017).
    fn reduce_field(self) -> String {
        match self {
            FuelOpCategory::Reduction(ReduceAxes::All) => "rall".to_string(),
            FuelOpCategory::Reduction(ReduceAxes::TrailingAxis) => "rlast".to_string(),
            FuelOpCategory::Reduction(ReduceAxes::Keepdim(m)) => format!("x{m:02x}"),
            _ => "-".to_string(),
        }
    }
}

/// Derive the KISS `sk4` `structure_key` token for a cell, independently of
/// Baracuda. `operands` are in canonical order — inputs then output
/// (§6.6-0014). Returns `None` (a typed decline, never a wrong token) on an
/// unmappable dtype, an empty operand list, a rank over `MAX_RANK` (8), more
/// than `MAX_OPERANDS` (8) operands, a malformed descriptor, a non-namespaced
/// target, or an invalid (negative) `gem` role extent.
pub fn derive_structure_key_token(
    op: FuelOpCategory,
    operands: &[FdxOperandDesc],
    target: &str,
) -> Option<String> {
    derive_structure_key_token_with_acc_mp(op, operands, target, None)
}

/// As [`derive_structure_key_token`], plus the sk4 non-contraction precision
/// coordinate (§6.7-0013).
///
/// Passing `None` is **not** "unknown" — it asserts the §6.17-0005 diagonal
/// (accumulator == compute dtype, `<mp>` at default), which is exactly what an
/// absent field means. That is why the plain entry point can delegate here
/// without changing the bytes of any token it produced before.
///
/// Declines (`None`) if an `(acc + mp)` is supplied for a `gem` cell: §6.7-0013
/// pins that a cell carries **at most one** precision field and that the two
/// never coexist, so emitting both would be invalid rather than merely odd.
pub fn derive_structure_key_token_with_acc_mp(
    op: FuelOpCategory,
    operands: &[FdxOperandDesc],
    target: &str,
    acc_mp: Option<AccMp>,
) -> Option<String> {
    if acc_mp.is_some() && matches!(op, FuelOpCategory::Contraction(_)) {
        return None;
    }
    let first = operands.first()?;
    if operands.len() > 8 {
        return None; // MAX_OPERANDS cap (§6.4-0002)
    }
    if operands
        .iter()
        .any(|o| o.shape.len() > 8 || o.shape.len() != o.strides.len())
    {
        return None; // MAX_RANK cap (§6.4-0001) / malformed descriptor
    }
    let dtype = dtype_token(first.dtype)?;
    if !target.contains(':') {
        return None; // namespaced target required (§6.8-0001)
    }

    // Iteration frame (§6.6-0013): rank = widest operand rank (§6.6-0006);
    // frame extent per axis = the maximum extent across the right-aligned
    // operands at that axis.
    let rank = operands.iter().map(|o| o.shape.len()).max().unwrap_or(0);
    let mut frame = vec![0i64; rank];
    for o in operands {
        let off = rank - o.shape.len();
        for (i, &e) in o.shape.iter().enumerate() {
            frame[off + i] = frame[off + i].max(e);
        }
    }

    // Field 4 — index width: max touched offset Σ|stride|·(ext−1) across
    // operands' own axes (§6.5-0011; a padded frame axis is stride-0 and
    // contributes 0).
    let max_touched: i128 = operands
        .iter()
        .map(|o| {
            o.strides
                .iter()
                .zip(o.shape.iter())
                .map(|(&s, &e)| i128::from(s.unsigned_abs()) * i128::from(e.max(1) - 1))
                .sum::<i128>()
        })
        .max()
        .unwrap_or(0);
    let index_width = if max_touched >= (1i128 << 31) {
        "ix64"
    } else {
        "ix32"
    };

    // Field 5 — work class: total element count of the ITERATION FRAME
    // (§6.5-0010) — the per-axis maximum extents, not operand 0's.
    let work_elems: i128 = frame.iter().map(|&e| i128::from(e)).product();
    let work_class = if work_elems <= 32 {
        "warp"
    } else if work_elems <= 1024 {
        "block"
    } else {
        "grid"
    };

    // §6.5-0009(b): every operand of a reduction cell whose reduced set
    // includes the innermost iteration-frame axis derives v1. Right-alignment
    // (§6.6-0013) maps every operand's innermost axis to frame axis rank−1,
    // so the gate is cell-level.
    let innermost_reduced = match op {
        FuelOpCategory::Reduction(ReduceAxes::All)
        | FuelOpCategory::Reduction(ReduceAxes::TrailingAxis) => true,
        FuelOpCategory::Reduction(ReduceAxes::Keepdim(m)) => {
            rank >= 1 && (m >> (rank - 1)) & 1 == 1
        }
        _ => false,
    };

    // Field 7 — per-operand sub-keys, canonical order (inputs then output,
    // §6.6-0014), each derived in the iteration frame.
    let operand_keys: Vec<String> = operands
        .iter()
        .map(|o| operand_sub_key(o, &frame, innermost_reduced))
        .collect();

    // Field 9 (gem only) — the sk4 contraction field (§6.7-0006).
    let contraction = match op {
        FuelOpCategory::Contraction(cell) => Some(contraction_field(&cell)?),
        _ => None,
    };

    let mut token = format!(
        "sk4|{op}|{dtype}|{target}|{idx}|{work}|r{rank}|{ops}|{reduce}",
        op = op.code(),
        idx = index_width,
        work = work_class,
        ops = operand_keys.join(";"),
        reduce = op.reduce_field(),
    );
    // §6.7-0013(c)/(e): the non-contraction precision field is
    // OMITTED-WHEN-ABSENT — not `-`, not empty. This is a deliberate contrast
    // with the MANDATORY reduce field above, which emits `-` when
    // inapplicable. Applying the reduce field's convention here would emit a
    // spurious trailing `|-` and fail the byte-match. The nearest precedent in
    // this very token teaches the opposite rule, which is what makes it a trap.
    if let Some(a) = acc_mp.filter(|a| a.deviates_from(first.dtype)) {
        token.push('|');
        token.push_str(&a.field()?);
    }
    if let Some(c) = contraction {
        token.push('|');
        token.push_str(&c);
    }
    Some(token)
}

/// The sk4 `gem` contraction field (§6.7-0006):
/// `c<m><n><k>/<kdiv>[/b<class>]/<wdt>/<acc>/<out>/<mp>` — six `/`-parts
/// non-batched, seven batched. Declines (`None`) on a negative role extent or
/// a dtype outside the closed §6.1 set, never guessing.
fn contraction_field(cell: &GemCell) -> Option<String> {
    let m = size_class(cell.m)?;
    let n = size_class(cell.n)?;
    let k = size_class(cell.k)?;
    let kdiv = div_bucket(cell.k);
    let batch = match cell.batch {
        Some(b) => format!("/b{}", size_class(b)?),
        None => String::new(),
    };
    let wdt = dtype_token(cell.weight_dtype)?;
    let acc = dtype_token(cell.acc_dtype)?;
    let out = dtype_token(cell.out_dtype)?;
    Some(format!(
        "c{m}{n}{k}/{kdiv}{batch}/{wdt}/{acc}/{out}/{mp}",
        mp = cell.math_precision.code(),
    ))
}

/// Contraction size class (§6.5-0008): `t` ≤ 8, `s` 9..=128, `m` 129..=2048,
/// `l` > 2048. A negative extent is invalid input → typed decline.
fn size_class(extent: i64) -> Option<char> {
    if extent < 0 {
        return None;
    }
    Some(if extent <= 8 {
        't'
    } else if extent <= 128 {
        's'
    } else if extent <= 2048 {
        'm'
    } else {
        'l'
    })
}

/// KISS-CLASSIFY §6.1 dtype token for a keyed dtype coordinate, over the
/// closed **24-token sk4** set (§6.1-0001).
///
/// FP8 tokens are width-prefixed and variant-explicit: Fuel's `F8E4M3` is the
/// OCP format → `f8e4m3fn`, and `F8E5M2` → `f8e5m2` **unsuffixed**, since only
/// the `fnuz` layouts deviate from IEEE E5M2. The reserved `fnuz` variants have
/// no Fuel `DType` and therefore can never be emitted here.
///
/// # §6.1-0001 has no site in Fuel at all — emitter OR parser
///
/// That clause binds a **reader of a `structure_key`**, and Fuel is not one: it
/// derives and emits keys, and treats foreign ones as opaque bytes (K1
/// opacity). Measured, not assumed — no split/decode path over a structure_key
/// exists anywhere outside test code, which parses only Fuel's own
/// just-emitted token to assert its shape.
///
/// An earlier version of this comment called the parse side "a real
/// conformance gap on the parse side, tracked separately". **That was false,
/// and correcting it is part of GAP-155**: an internal type enum recognizes
/// nothing, and Fuel's internal dtype vocabulary (`f8e4m3`) is not the seam
/// spelling (`f8e4m3fn`) a reader would ever see — so no reserved token can
/// reach `DType::from_str` from inside Fuel. Fuel's KISS-Classify conformance
/// claim is scoped to the **emit** surface.
///
/// The reserved-token declines that do exist — `fuel_ir::RESERVED_DTYPE_TOKENS`
/// and `FkcError::ReservedScalarType` on hand-authored `.fkc.md` input — are
/// diagnostics on Fuel's own ingest surfaces, NOT §6.1-0001 conformance.
///
/// The two 8-bit MX **scales** (`F8E8M0`/`F8E6M2`) entered the closed set at
/// sk4 and are emitted. Note they are dtype-bearing here only via
/// `operands.first()` or a `gem` precision coordinate: a scale riding as a
/// *sibling* operand never reaches a dtype position at all, because non-first
/// operands contribute layout only.
///
/// The block-scoped sub-byte **element** formats (`F6E2M3`/`F6E3M2`/`F4`) are
/// still outside the set and typed-decline (`None`), never a guessed token.
///
/// **Declining here aborts the WHOLE derivation** (`?` at the call site), not
/// just this field — so a dtype moving between the emitted and declined sets
/// changes whether a cell emits a structure key at all.
///
/// **Delegates to [`fuel_ir::sk4_token`], the single source of the seam
/// spelling.** This function used to carry its own copy of the table.
///
/// Two hand-maintained copies of one mapping is the drift pattern this repo
/// keeps re-finding, and it is worse than usual here: the copy that stops
/// being edited does not fail loudly, it emits a **retired spelling under a
/// current version prefix** — which §6.1-0004 forbids outright, and which no
/// test downstream of this function can detect, because the key it produces is
/// still well-formed.
///
/// The exhaustiveness anchor moved with the table rather than being lost:
/// `sk4_token` is a wildcard-free `match`, so a new Fuel `DType` is still a
/// compile error forcing a decision — now in the crate that owns `DType`,
/// where the emitted token is additionally checked against the closed
/// vocabulary it has to belong to.
fn dtype_token(dt: DType) -> Option<&'static str> {
    fuel_ir::sk4_token(dt)
}

/// One operand's `<contig>/<bcasthex>/<vec>/<div>/<flip>` sub-key (§6.6-0007),
/// derived in the iteration frame (§6.6-0013): a rank-deficient operand is
/// right-aligned, with every frame axis below `rank − r` treated as broadcast
/// (stride 0) for it.
fn operand_sub_key(o: &FdxOperandDesc, frame: &[i64], innermost_reduced: bool) -> String {
    let rank = frame.len();
    let off = rank - o.shape.len();

    // The padded (frame-aligned) view: padded axes carry the frame extent
    // with stride 0.
    let ext_p: Vec<i64> = (0..rank)
        .map(|i| if i < off { frame[i] } else { o.shape[i - off] })
        .collect();
    let str_p: Vec<i64> = (0..rank)
        .map(|i| if i < off { 0 } else { o.strides[i - off] })
        .collect();

    // Broadcast-axis mask (§6.6-0008): bit i set iff iteration-frame axis i
    // has extent > 1 and this operand's stride along it is 0.
    let mut mask = 0u8;
    for i in 0..rank.min(8) {
        if frame[i] > 1 && str_p[i] == 0 {
            mask |= 1 << i;
        }
    }

    let layout = layout_code(&ext_p, &str_p);

    // Own innermost axis (§6.3-0011): axis rank−1 of the operand's OWN shape
    // (right-aligned to the frame innermost). A rank-0 operand has none.
    let (inner_extent, inner_stride) = match o.shape.len().checked_sub(1) {
        Some(i) => (o.shape[i], Some(o.strides[i])),
        None => (1, None),
    };

    let div = div_bucket(inner_extent);

    // Vector-access width (§6.5-0009 / §6.5-0013): v1 on a broadcast layout or
    // any broadcast-marked axis, on a reduced innermost axis of a `red` cell,
    // on a missing/sub-byte/unaligned base, or a non-forward-unit inner
    // stride; else the largest L ∈ {8,4,2} within the 16-byte cap whose exact
    // modulo divides the alignment and the inner extent.
    let dsz = o.dtype.size_in_bytes();
    let vec = if layout == "br"
        || mask != 0
        || innermost_reduced
        || dsz == 0
        || o.align_bytes == 0
        || inner_stride != Some(1)
    {
        "v1"
    } else {
        let mut picked = "v1";
        for &l in &[8i64, 4, 2] {
            let vbytes = (l as u32) * (dsz as u32);
            // `inner_extent >= l` carries the same `E >= N` guard `div_bucket`
            // uses (§6.5-0012): without it the `inner_extent % l == 0` test is
            // VACUOUSLY true at E=0 (every L divides 0), mis-deriving v4 for an
            // empty run — the §6.5-0009(c) zero-extent trap (KISS #82 F4 / #87).
            if vbytes <= 16
                && o.align_bytes.is_multiple_of(vbytes)
                && inner_extent >= l
                && inner_extent % l == 0
            {
                picked = match l {
                    8 => "v8",
                    4 => "v4",
                    _ => "v2",
                };
                break;
            }
        }
        picked
    };

    format!(
        "{layout}/{mask:02x}/{vec}/{div}/{flip}",
        flip = if o.flipped { "r" } else { "f" },
    )
}

/// Layout tag (§6.5-0002), the pinned 4-step algorithm over `|stride|`,
/// active non-unit axes, innermost first: **(1)** `br` if any axis of
/// extent > 1 has stride 0; **(2)** `co` if each active non-unit axis's
/// `|stride|` equals the running product of the inner active non-unit
/// extents (a fully reversed view is therefore `co` — the reversal lives in
/// the flipped flag); **(3)** `ic` if the innermost active non-unit axis has
/// `|stride| == 1`; **(4)** else `st`. No active axis of extent > 1 ⇒ `co`
/// (empty product).
fn layout_code(ext: &[i64], strides: &[i64]) -> &'static str {
    if ext.iter().zip(strides).any(|(&e, &s)| e > 1 && s == 0) {
        return "br";
    }
    let mut p: i128 = 1;
    let mut contiguous = true;
    for i in (0..ext.len()).rev() {
        let e = ext[i];
        if e <= 1 {
            continue; // unit / zero-extent axes are excluded from the product
        }
        if i128::from(strides[i].unsigned_abs()) != p {
            contiguous = false;
            break;
        }
        p *= i128::from(e);
    }
    if contiguous {
        return "co";
    }
    if let Some(i) = (0..ext.len()).rev().find(|&i| ext[i] > 1)
        && strides[i].unsigned_abs() == 1
    {
        return "ic";
    }
    "st"
}

/// Inner-extent divisibility bucket (§6.5-0012), with the pinned `E ≥ N`
/// guard: `d16` iff `E ≥ 16 ∧ 16|E`; else `d8`, `d4`, `d2` likewise; else
/// `da` — covering odd `E`, `E = 1`, and `E = 0` (the zero-extent trap: a
/// guardless `E mod 16 == 0` would mis-bucket `E = 0` as `d16`).
fn div_bucket(e: i64) -> &'static str {
    if e >= 16 && e % 16 == 0 {
        "d16"
    } else if e >= 8 && e % 8 == 0 {
        "d8"
    } else if e >= 4 && e % 4 == 0 {
        "d4"
    } else if e >= 2 && e % 2 == 0 {
        "d2"
    } else {
        "da"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::{DType, Layout, Shape, StrideVec};

    fn co(dims: &[usize], dtype: DType) -> FdxOperandDesc {
        FdxOperandDesc::from_layout(&Layout::contiguous(Shape::from_dims(dims)), dtype)
    }
    fn f32c(dims: &[usize]) -> FdxOperandDesc {
        co(dims, DType::F32)
    }

    /// **SPELLED and DERIVABLE are different axes, and only one of them was ever
    /// measured.** `fuel_ir::sk4_token` maps a `DType` to its seam token — that is
    /// *spelling*, and `token_kind.rs` already asserts a 15/7/2 partition over it.
    /// This asks the other question: **can Fuel construct a real cell whose operand 0
    /// carries that dtype, so a token containing the spelling actually comes out?**
    ///
    /// Written because a cross-project record carried three dtypes (`i16`, `f8e8m0`,
    /// `f8e6m2`) as *"legal token but not derivable"* — a Fuel-side capability gap.
    /// The claim conflated the axes: all three are spelled (`token_kind.rs:82,89,90`).
    /// Rather than answer for those three and leave the rest assumed, this measures
    /// **every** dtype that has a spelling, so the answer cannot rot into a claim
    /// about a subset someone happened to ask about.
    ///
    /// The assertion is `spelled ⇒ derivable`, and it is deliberately exhaustive over
    /// `DType::ALL`: a future dtype that can be spelled but not built at operand 0
    /// fails **here**, with its name, instead of being discovered by a peer.
    #[test]
    fn every_spelled_dtype_is_also_derivable_at_operand_zero() {
        let mut spelled = 0usize;
        let mut not_derivable: Vec<(DType, &'static str)> = Vec::new();
        let mut mismatched: Vec<(DType, &'static str, String)> = Vec::new();

        for dt in DType::ALL.iter().copied() {
            let Some(spelling) = fuel_ir::sk4_token(dt) else {
                continue; // no seam token — outside this axis by construction
            };
            spelled += 1;

            // The simplest cell that puts a dtype at operand 0.
            let ops = [co(&[4096], dt), co(&[4096], dt)];
            match derive_structure_key_token(FuelOpCategory::BinaryElementwise, &ops, "cuda:sm89") {
                None => not_derivable.push((dt, spelling)),
                Some(token) => {
                    // Field 3 (0-indexed 2) is `<dtype>`: sk4|<op>|<dtype>|<target>|…
                    let got = token.split('|').nth(2).unwrap_or("<missing>");
                    if got != spelling {
                        mismatched.push((dt, spelling, got.to_string()));
                    }
                }
            }
        }

        // Non-vacuity: if `sk4_token` ever returned `None` for everything, the loop
        // above would pass by examining nothing. 15 is the measured supported count
        // that `token_kind.rs`'s partition test pins independently.
        assert_eq!(
            spelled, 15,
            "expected 15 spelled dtypes (token_kind.rs pins this); got {spelled} — \
             this test would be vacuous at 0",
        );
        assert!(
            not_derivable.is_empty(),
            "spelled but NOT derivable at operand 0: {not_derivable:?} — \
             a dtype with a seam token that no cell can carry is a real capability gap",
        );
        assert!(
            mismatched.is_empty(),
            "derived token's dtype field disagrees with the spelling: {mismatched:?}",
        );
    }

    /// A bit-stable non-batched f32 gem cell with the given role extents.
    fn gem_f32(m: i64, n: i64, k: i64) -> GemCell {
        GemCell {
            m,
            n,
            k,
            batch: None,
            weight_dtype: DType::F32,
            acc_dtype: DType::F32,
            out_dtype: DType::F32,
            math_precision: GemMathPrecision::BitStable,
        }
    }

    // ---- (a) sk4 prefix on every token class --------------------------------

    /// The relu_add f32 grid-stride freeze-gate cell (condition-1): 3 rank-1
    /// f32 operands [4096], contiguous, offset 0 (align 256): in0, in1, out.
    /// Byte-for-byte the KISS PR #81 staged golden
    /// (`relu_add_generated_r1_cell`).
    #[test]
    fn fuel_derives_relu_add_sk4_token_byte_for_byte() {
        let op = f32c(&[4096]);
        let token = derive_structure_key_token(
            FuelOpCategory::BinaryElementwise,
            &[op.clone(), op.clone(), op],
            "cuda:sm89",
        )
        .expect("relu_add f32 must derive a token");
        assert_eq!(
            token,
            "sk4|bin|f32|cuda:sm89|ix32|grid|r1|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-"
        );
    }

    /// Every derivable op-family token carries the sk4 prefix and no sk2 bytes.
    #[test]
    fn sk4_prefix_on_every_token_class() {
        let op = f32c(&[4096]);
        let cats = [
            FuelOpCategory::BinaryElementwise,
            FuelOpCategory::TernaryElementwise,
            FuelOpCategory::Reduction(ReduceAxes::All),
            FuelOpCategory::Reduction(ReduceAxes::TrailingAxis),
            FuelOpCategory::Reduction(ReduceAxes::Keepdim(0x02)),
            FuelOpCategory::Contraction(gem_f32(8, 4096, 4096)),
            FuelOpCategory::Normalization,
            FuelOpCategory::Convolution,
            FuelOpCategory::Pooling,
            FuelOpCategory::Indexing,
            FuelOpCategory::ShapeLayout,
            FuelOpCategory::Sorting,
            FuelOpCategory::Fft,
            FuelOpCategory::Linalg,
            FuelOpCategory::Random,
            FuelOpCategory::SegmentOps,
            FuelOpCategory::Softmax,
            FuelOpCategory::Attention,
            FuelOpCategory::Loss,
        ];
        for cat in cats {
            let token = derive_structure_key_token(cat, std::slice::from_ref(&op), "cuda:sm89")
                .unwrap_or_else(|| panic!("{:?} must derive", cat));
            assert!(token.starts_with("sk4|"), "{token} lacks the sk4 prefix");
            assert!(!token.contains("sk2"), "{token} carries sk2 bytes");
        }
    }

    // ---- (e) non-gem tokens byte-identical to sk2 modulo prefix -------------

    /// The four committed sk2-era battery cells re-derive with ONLY the prefix
    /// changed (the sk2 tokens are pinned inline from the fdc1e987/97307020
    /// test battery).
    #[test]
    fn non_gem_tokens_byte_identical_to_sk2_modulo_prefix() {
        let cases: [(FuelOpCategory, Vec<FdxOperandDesc>, &str); 4] = [
            (
                FuelOpCategory::BinaryElementwise,
                vec![f32c(&[4096]), f32c(&[4096]), f32c(&[4096])],
                // sk2: "sk2|bin|f32|cuda:sm89|ix32|grid|r1|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-"
                "sk4|bin|f32|cuda:sm89|ix32|grid|r1|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-",
            ),
            (
                FuelOpCategory::BinaryElementwise,
                vec![f32c(&[7])],
                // sk2: "sk2|bin|f32|cuda:sm89|ix32|warp|r1|co/00/v1/da/f|-"
                "sk4|bin|f32|cuda:sm89|ix32|warp|r1|co/00/v1/da/f|-",
            ),
            (
                FuelOpCategory::BinaryElementwise,
                vec![co(&[4096], DType::I16)],
                // sk2: "sk2|bin|s16|cuda:sm89|ix32|grid|r1|co/00/v8/d16/f|-"
                "sk4|bin|i16|cuda:sm89|ix32|grid|r1|co/00/v8/d16/f|-",
            ),
            (
                FuelOpCategory::BinaryElementwise,
                vec![f32c(&[128, 256])],
                // sk2: "sk2|bin|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f|-"
                "sk4|bin|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f|-",
            ),
        ];
        for (cat, ops, expect) in cases {
            let token = derive_structure_key_token(cat, &ops, "cuda:sm89").expect("derives");
            assert_eq!(token, expect);
        }
    }

    // ---- (b) the sk4 gem 6/7-component contraction group --------------------

    /// KISS Appendix A.1 dense GEMM skinny-decode cell
    /// `[8,4096]·[4096,4096]→[8,4096]`, f32, non-batched, bit-stable — the sk4
    /// precision group is `/f32/f32/f32/st`. Byte-for-byte the staged golden
    /// (`a1_dense_contraction_cuda` / `a1_dense_contraction_vulkan_target`).
    #[test]
    fn kiss_a1_gem_skinny_decode_golden() {
        let ops = [f32c(&[8, 4096]), f32c(&[4096, 4096]), f32c(&[8, 4096])];
        let cell = FuelOpCategory::Contraction(gem_f32(8, 4096, 4096));
        let cuda = derive_structure_key_token(cell, &ops, "cuda:sm89").expect("derives");
        assert_eq!(
            cuda,
            "sk4|gem|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/f32/f32/f32/st"
        );
        // The same cell for a Vulkan target is a different cell (byte-exact
        // target rule, §6.8-0002).
        let vk = derive_structure_key_token(cell, &ops, "vulkan:spirv1.6").expect("derives");
        assert_eq!(
            vk,
            "sk4|gem|f32|vulkan:spirv1.6|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/f32/f32/f32/st"
        );
    }

    /// The namespaced target is a **byte-exact passthrough coordinate**
    /// (§6.8-0002): whatever the caller supplies appears as field 4 of the
    /// token, character for character — no normalization, no reordering, and
    /// **no assumption about how many `.`-separated fields it has**.
    ///
    /// # Why a property, when goldens already cover this
    ///
    /// The goldens above pin *specific vocabulary spellings*, and those
    /// spellings belong to a vocabulary **Fuel does not own**. KISS #200 takes
    /// `vulkan:` from four fields to five (cooperative *vector* is structurally
    /// unmergeable with cooperative *matrix*, so it cannot share the `<coop>`
    /// tuple grammar), which changes the bytes of **every** `vulkan:` token —
    /// including the four-field `vulkan:sg64.ops-abr.arith-f16.cm-none` this
    /// crate's byte-match corpus pins. A golden per vocabulary version means a
    /// Fuel-side edit per foreign vocabulary cut, forever.
    ///
    /// This test asserts the property those goldens are *evidence for*, so it
    /// is invariant under every future vocabulary version. The v4 five-field
    /// shape is exercised below **before it exists upstream** — that case is
    /// the reason this test was written and it is not reachable from any
    /// current corpus.
    ///
    /// # Why the goldens are KEPT rather than replaced
    ///
    /// A property test is **self-authored**: it proves the class while
    /// comparing this implementation against itself, so it would accept a
    /// *wrong* passthrough that still round-trips. The corpus goldens are the
    /// only assertions here whose expected value has an author who is not
    /// Fuel. Strengthening the class check must not trade away the single
    /// cross-authored comparison, so this is strictly additive.
    #[test]
    fn namespaced_target_is_a_byte_exact_passthrough_for_any_vocabulary() {
        let ops = [f32c(&[8, 4096]), f32c(&[4096, 4096]), f32c(&[8, 4096])];
        let cell = FuelOpCategory::Contraction(gem_f32(8, 4096, 4096));

        // Deliberately spans: the two spellings the goldens pin, the four-field
        // v3 capability set the corpus hardcodes, the **v4 five-field shape
        // that does not exist yet**, and a namespace Fuel has never seen whose
        // mixed case and separators would be destroyed by any normalization.
        let targets = [
            "cuda:sm89",
            "vulkan:spirv1.6",
            "vulkan:sg64.ops-abr.arith-f16.cm-none",
            "vulkan:sg64.ops-abr.arith-f16.cm-none.cv-none",
            "zzfuture:A-b.C_9.MixedCase.trailing-Z",
        ];

        for target in targets {
            let token = derive_structure_key_token(cell, &ops, target)
                .unwrap_or_else(|| panic!("`{target}` is namespaced and must derive"));
            let fields: Vec<&str> = token.split('|').collect();
            // Non-vacuity: a token too short to *have* a target field would
            // otherwise let the equality below pass over `None`-ish nonsense.
            assert!(
                fields.len() > 3,
                "`{target}`: token has {} fields, no target coordinate to check: {token}",
                fields.len()
            );
            assert_eq!(
                fields[3], target,
                "target must survive byte-exact as field 4 (§6.8-0002); token: {token}"
            );
        }

        // Negative control, and it is what makes the loop above mean something:
        // the *only* thing admitting these strings is the namespace separator
        // (§6.8-0001). Strip the `:` from a target that just passed and the
        // deriver must decline — otherwise "it passed for every input" would be
        // consistent with the target field being ignored entirely.
        assert_eq!(
            derive_structure_key_token(cell, &ops, "vulkansg64.ops-abr.arith-f16.cm-none"),
            None,
            "a non-namespaced target must decline, not pass through"
        );
    }

    /// A batched gem cell carries the conditionally-present `b<class>` right
    /// after `<kdiv>` (7 `/`-parts); the non-batched twin omits it entirely
    /// (6 parts). Byte-for-byte the staged `sk4_gem_batched_cell` golden.
    #[test]
    fn sk4_gem_batched_cell_golden() {
        let ops = [f32c(&[256, 4096]), f32c(&[4096, 4096]), f32c(&[256, 4096])];
        let batched = GemCell {
            batch: Some(256),
            ..gem_f32(256, 4096, 4096)
        };
        let token =
            derive_structure_key_token(FuelOpCategory::Contraction(batched), &ops, "cuda:sm90")
                .expect("derives");
        assert_eq!(
            token,
            "sk4|gem|f32|cuda:sm90|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|cmll/d16/bm/f32/f32/f32/st"
        );
        // The non-batched twin differs exactly by the absent /bm coordinate.
        let plain = derive_structure_key_token(
            FuelOpCategory::Contraction(gem_f32(256, 4096, 4096)),
            &ops,
            "cuda:sm90",
        )
        .expect("derives");
        assert_eq!(plain, token.replace("/bm/", "/"));
    }

    /// SIMT-f32 (`st`) and TF32 (`rm`) are the same shape but distinct cells:
    /// the `<mp>` coordinate distinguishes them (the spec-forbidden `f32s`
    /// dtype hack is retired). Byte-for-byte the staged
    /// `sk4_simt_f32_vs_tf32_distinct_by_mp` goldens.
    #[test]
    fn sk4_gem_simt_f32_vs_tf32_distinct_by_mp() {
        let ops = [f32c(&[8, 4096]), f32c(&[4096, 4096]), f32c(&[8, 4096])];
        let simt = derive_structure_key_token(
            FuelOpCategory::Contraction(gem_f32(8, 4096, 4096)),
            &ops,
            "cuda:sm90",
        )
        .expect("derives");
        let tf32_cell = GemCell {
            math_precision: GemMathPrecision::ReducedMantissa,
            ..gem_f32(8, 4096, 4096)
        };
        let tf32 =
            derive_structure_key_token(FuelOpCategory::Contraction(tf32_cell), &ops, "cuda:sm90")
                .expect("derives");
        assert_eq!(
            simt,
            "sk4|gem|f32|cuda:sm90|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/f32/f32/f32/st"
        );
        assert_eq!(
            tf32,
            "sk4|gem|f32|cuda:sm90|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/f32/f32/f32/rm"
        );
        assert_ne!(simt, tf32, "SIMT-f32 and TF32 must not collide");
    }

    /// The mixed-precision FP8 cell the sk4 bump exists to disambiguate,
    /// with the variant-explicit `e4m3fn` spelling in BOTH the primary and
    /// the weight coordinate. Byte-for-byte the staged
    /// `sk4_mixed_precision_fp8_disambiguated` golden (its second, fully
    /// Fuel-representable vector: E4M3×E4M3→F16, f32 acc, bit-stable; the
    /// e5m2-weight first vector is not derivable — Fuel's `DType` carries no
    /// `e5m2` storage dtype).
    #[test]
    fn sk4_gem_mixed_precision_fp8_golden() {
        // f8 operands at a 4-byte-aligned view (start_offset 4 → align 4) so
        // the 1-byte dtype derives v4 (matching the staged golden's sub-keys),
        // not the offset-0 v8.
        let f8 = |dims: &[usize]| {
            FdxOperandDesc::from_layout(
                &Layout::new(
                    Shape::from_dims(dims),
                    Layout::contiguous(Shape::from_dims(dims))
                        .stride()
                        .iter()
                        .copied()
                        .collect::<StrideVec>(),
                    4,
                ),
                DType::F8E4M3,
            )
        };
        let ops = [f8(&[8, 4096]), f8(&[4096, 4096]), f8(&[8, 4096])];
        let cell = GemCell {
            weight_dtype: DType::F8E4M3,
            acc_dtype: DType::F32,
            out_dtype: DType::F16,
            ..gem_f32(8, 4096, 4096)
        };
        let token =
            derive_structure_key_token(FuelOpCategory::Contraction(cell), &ops, "cuda:sm90")
                .expect("derives");
        assert_eq!(
            token,
            "sk4|gem|f8e4m3fn|cuda:sm90|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/f8e4m3fn/f32/f16/st"
        );
        // The f32-out twin is a DISTINCT token (the sk2 collision resolved
        // in-key by the precision coordinates, §6.6-0018).
        let twin = GemCell {
            out_dtype: DType::F32,
            ..cell
        };
        let twin_token =
            derive_structure_key_token(FuelOpCategory::Contraction(twin), &ops, "cuda:sm90")
                .expect("derives");
        assert_ne!(
            token, twin_token,
            "mixed-precision FP8 cells must not collide under sk4"
        );
    }

    /// A gem cell declines (never guesses) on a precision coordinate outside
    /// the closed §6.1 set (an MX dtype) or an invalid negative role extent.
    #[test]
    fn gem_declines_unmappable_or_invalid() {
        let ops = [f32c(&[8, 4096]), f32c(&[4096, 4096]), f32c(&[8, 4096])];
        let mx_weight = GemCell {
            weight_dtype: DType::F4,
            ..gem_f32(8, 4096, 4096)
        };
        assert_eq!(
            derive_structure_key_token(FuelOpCategory::Contraction(mx_weight), &ops, "cuda:sm89"),
            None
        );
        let negative_m = gem_f32(-1, 4096, 4096);
        assert_eq!(
            derive_structure_key_token(FuelOpCategory::Contraction(negative_m), &ops, "cuda:sm89"),
            None
        );
        let negative_batch = GemCell {
            batch: Some(-2),
            ..gem_f32(8, 4096, 4096)
        };
        assert_eq!(
            derive_structure_key_token(
                FuelOpCategory::Contraction(negative_batch),
                &ops,
                "cuda:sm89"
            ),
            None
        );
    }

    /// The full contraction size-class ladder (§6.5-0008): t ≤8, s 9..=128,
    /// m 129..=2048, l >2048, and the K-divisibility bucket rides K.
    #[test]
    fn gem_size_class_ladder() {
        let ops = [f32c(&[8, 4096]), f32c(&[4096, 4096]), f32c(&[8, 4096])];
        for (m, n, k, expect) in [
            (8i64, 9, 129, "ctsm/da"),    // K=129 odd → da
            (128, 2048, 2049, "csml/da"), // K=2049 odd → da
            (1, 129, 24, "ctms/d8"),      // K=24 → d8 (mod 16 = 8)
            (3000, 8, 4096, "cltl/d16"),
        ] {
            let token = derive_structure_key_token(
                FuelOpCategory::Contraction(gem_f32(m, n, k)),
                &ops,
                "cuda:sm89",
            )
            .expect("derives");
            let field = token.rsplit('|').next().unwrap();
            assert!(
                field.starts_with(expect),
                "gem({m},{n},{k}) contraction field {field} != {expect}…"
            );
        }
    }

    // ---- (c)/(d) retired + reserved spellings -------------------------------

    /// No derivable token — any dtype in the primary or gem precision
    /// positions — ever contains a retired (`f32s`, bare `e4m3`) or reserved
    /// (`fnuz`) spelling; `F8E4M3` spells `e4m3fn`. Exhaustive over Fuel's
    /// `DType`. (The deriver is emit-only: it has NO token parse path, so the
    /// fnuz reserved-on-parse typed-decline lives with the readers — here the
    /// reserved spellings are unrepresentable by construction.)
    #[test]
    fn retired_and_reserved_spellings_never_emitted() {
        const ALL: [DType; 17] = [
            DType::U8,
            DType::I8,
            DType::U32,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::BF16,
            DType::F16,
            DType::F32,
            DType::F64,
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F6E2M3,
            DType::F6E3M2,
            DType::F4,
            DType::F8E8M0,
            DType::F8E6M2,
        ];
        let assert_clean = |token: &str| {
            assert!(!token.contains("f32s"), "retired f32s spelling in {token}");
            assert!(!token.contains("fnuz"), "reserved fnuz spelling in {token}");
            assert!(
                !token.contains("|e4m3|"),
                "retired bare e4m3 (primary) in {token}"
            );
            assert!(
                !token.contains("/e4m3/"),
                "retired bare e4m3 (gem group) in {token}"
            );
            // sk4 retires the UNPREFIXED fp8 spellings too. The delimiters are
            // load-bearing: `f8e4m3fn` CONTAINS `e4m3fn` as a substring, so a
            // bare `contains("e4m3fn")` would reject the CORRECT sk4 token.
            // `/f8e4m3fn/` does not match `/e4m3fn/`, so the delimited form
            // catches only the retired spelling.
            assert!(
                !token.contains("|e4m3fn|"),
                "retired unprefixed e4m3fn (primary) in {token}"
            );
            assert!(
                !token.contains("/e4m3fn/"),
                "retired unprefixed e4m3fn (gem group) in {token}"
            );
            assert!(
                !token.contains("|e5m2|"),
                "retired unprefixed e5m2 (primary) in {token}"
            );
            assert!(
                !token.contains("/e5m2/"),
                "retired unprefixed e5m2 (gem group) in {token}"
            );
            assert!(
                !token.contains("|s8|") && !token.contains("/s8/"),
                "retired sk3 s8 spelling in {token}"
            );
            assert!(
                !token.contains("|s16|") && !token.contains("/s16/"),
                "retired sk3 s16 spelling in {token}"
            );
        };
        for dt in ALL {
            // Primary position (non-gem).
            if let Some(token) = derive_structure_key_token(
                FuelOpCategory::BinaryElementwise,
                &[co(&[4096], dt)],
                "cuda:sm89",
            ) {
                assert_clean(&token);
            }
            // Every gem precision position at once.
            let cell = GemCell {
                weight_dtype: dt,
                acc_dtype: dt,
                out_dtype: dt,
                ..gem_f32(8, 4096, 4096)
            };
            if let Some(token) = derive_structure_key_token(
                FuelOpCategory::Contraction(cell),
                &[
                    co(&[8, 4096], dt),
                    co(&[4096, 4096], dt),
                    co(&[8, 4096], dt),
                ],
                "cuda:sm89",
            ) {
                assert_clean(&token);
            }
        }
        // The OCP FP8 dtype spells f8e4m3fn (f8 width prefix), in both positions.
        let token = derive_structure_key_token(
            FuelOpCategory::BinaryElementwise,
            &[co(&[4096], DType::F8E4M3)],
            "cuda:sm89",
        )
        .expect("f8e4m3fn derives");
        assert!(token.starts_with("sk4|bin|f8e4m3fn|"), "got {token}");
    }

    /// **The behavioural half of the sk3→sk4 regen**, and the reason a
    /// spelling-only test is not sufficient.
    ///
    /// `dtype_token`'s `None` is consumed by a `?` on the WHOLE derivation, not
    /// as a per-field fallback. So moving the two 8-bit MX scales from the
    /// declined set into the emitted set does not merely change a token in an
    /// existing key — it changes cells that previously produced **no structure
    /// key at all** into cells that produce one. Downstream, telemetry rows
    /// that never existed begin existing.
    ///
    /// A test that only checked the new spellings appear where old ones did
    /// would pass without ever exercising that, because those cells emit
    /// nothing to inspect before the change.
    ///
    /// Asserted in BOTH directions: the scales now emit a well-formed key, and
    /// the block-scoped sub-byte ELEMENTS still decline. Without the second
    /// half this would also pass if the decline set had been emptied entirely.
    #[test]
    fn sk4_mx_scales_go_from_silent_to_emitting() {
        // Previously silent (whole-derivation decline), now emitting.
        for (dt, expected) in [(DType::F8E8M0, "f8e8m0"), (DType::F8E6M2, "f8e6m2")] {
            let token = derive_structure_key_token(
                FuelOpCategory::BinaryElementwise,
                &[co(&[4096], dt)],
                "cuda:sm89",
            )
            .unwrap_or_else(|| {
                panic!("{dt:?} is in the sk4 §6.1 set and must now derive a key, not decline")
            });

            let parts: Vec<&str> = token.split('|').collect();
            assert_eq!(
                parts.len(),
                9,
                "non-gem sk4 key must have 9 `|`-fields, got {} in {token}",
                parts.len(),
            );
            assert_eq!(parts[0], "sk4", "wrong schema prefix in {token}");
            assert_eq!(parts[2], expected, "wrong §6.1 dtype token in {token}");
        }

        // Negative control: the block-scoped sub-byte ELEMENT formats are still
        // outside the closed set and must still decline. If this half is ever
        // removed, the test above passes for a deriver that emits everything.
        for dt in [DType::F6E2M3, DType::F6E3M2, DType::F4] {
            assert_eq!(
                derive_structure_key_token(
                    FuelOpCategory::BinaryElementwise,
                    &[co(&[4096], dt)],
                    "cuda:sm89",
                ),
                None,
                "{dt:?} is NOT in the sk4 §6.1 set and must still typed-decline",
            );
        }
    }

    // ---- (acc + mp), the sk4 non-contraction precision field ---------------

    /// §6.7-0013(c)/(d): when NEITHER coordinate deviates, the field is omitted
    /// **entirely** — and the resulting token must be byte-identical to the one
    /// derived without any `(acc + mp)` at all.
    ///
    /// This is the **additivity lock**, asserted rather than cited. §6.7-0013
    /// claims byte-stability against the pre-sk4 codec; a byte-identity claim
    /// between two codecs is exactly the kind that reads true and isn't, so it
    /// is derived here instead of trusted. It is also what makes the field
    /// additive in KISS's strict sense — *no previously-derivable token is
    /// affected by its absence* — rather than merely appended.
    #[test]
    fn acc_mp_omitted_entirely_when_nothing_deviates() {
        let ops = [co(&[4096], DType::F32)];
        let plain = derive_structure_key_token(
            FuelOpCategory::Reduction(ReduceAxes::All),
            &ops,
            "cuda:sm89",
        )
        .expect("plain reduction derives");

        // acc == compute dtype, mp == default ⇒ nothing to declare.
        let diagonal = derive_structure_key_token_with_acc_mp(
            FuelOpCategory::Reduction(ReduceAxes::All),
            &ops,
            "cuda:sm89",
            Some(AccMp {
                acc_dtype: DType::F32,
                math_precision: GemMathPrecision::BitStable,
            }),
        )
        .expect("diagonal reduction derives");

        assert_eq!(
            diagonal, plain,
            "a non-deviating (acc+mp) must produce a BYTE-IDENTICAL token; any difference means the field is not additive and every token a consumer already holds has silently changed meaning",
        );
        // §6.7-0013(e): omitted-when-absent, NOT the reduce field's `-`.
        assert!(
            !plain.ends_with("|-|-") && !diagonal.ends_with("|-|-"),
            "a spurious trailing `|-` means the reduce field's MANDATORY convention was applied to an OMITTED-WHEN-ABSENT field: {diagonal}",
        );
    }

    /// §6.7-0013(a)/(b): emitted iff a coordinate deviates, and when emitted
    /// **both** slots are spelled — including one sitting at its default.
    #[test]
    fn acc_mp_emitted_with_both_slots_when_either_deviates() {
        let ops = [co(&[4096], DType::F16)];
        let derive = |a: AccMp| {
            derive_structure_key_token_with_acc_mp(
                FuelOpCategory::Reduction(ReduceAxes::All),
                &ops,
                "cuda:sm89",
                Some(a),
            )
            .expect("derives")
        };

        // (i) accumulator deviates; mp sits at its default and is STILL spelled.
        let acc_only = derive(AccMp {
            acc_dtype: DType::F32,
            math_precision: GemMathPrecision::BitStable,
        });
        assert!(acc_only.ends_with("|f32/st"), "got {acc_only}");

        // (ii) mp deviates; accumulator equals compute dtype and is STILL spelled.
        let mp_only = derive(AccMp {
            acc_dtype: DType::F16,
            math_precision: GemMathPrecision::ReducedMantissa,
        });
        assert!(mp_only.ends_with("|f16/rm"), "got {mp_only}");

        // Non-vacuity: the two deviating forms differ from each other AND from
        // the diagonal, so the field is actually discriminating cells.
        let diagonal = derive(AccMp {
            acc_dtype: DType::F16,
            math_precision: GemMathPrecision::BitStable,
        });
        assert_ne!(acc_only, diagonal);
        assert_ne!(mp_only, diagonal);
        assert_ne!(acc_only, mp_only);

        // The field is the LAST `|`-part, i.e. the optional-trailing slot.
        assert_eq!(
            acc_only.split('|').count(),
            10,
            "9 base fields + 1: {acc_only}"
        );
    }

    /// §6.7-0013: a cell carries **at most one** precision field, and the
    /// contraction group and `(acc + mp)` **never coexist**. Supplying both is
    /// invalid, so the deriver declines rather than emitting an 11-field token
    /// that no reader's 9-or-10 dispatch could parse.
    #[test]
    fn acc_mp_on_a_gem_cell_declines_rather_than_coexisting() {
        let cell = gem_f32(8, 4096, 4096);
        let ops = [
            co(&[8, 4096], DType::F32),
            co(&[4096, 4096], DType::F32),
            co(&[8, 4096], DType::F32),
        ];
        // Control: the same gem cell derives fine with no (acc+mp).
        assert!(
            derive_structure_key_token(FuelOpCategory::Contraction(cell), &ops, "cuda:sm89")
                .is_some(),
            "control: the gem cell must derive without an (acc+mp), or the decline below proves nothing about coexistence",
        );
        assert_eq!(
            derive_structure_key_token_with_acc_mp(
                FuelOpCategory::Contraction(cell),
                &ops,
                "cuda:sm89",
                Some(AccMp {
                    acc_dtype: DType::F32,
                    math_precision: GemMathPrecision::ReducedMantissa
                }),
            ),
            None,
            "a gem cell carrying BOTH precision fields must decline",
        );
    }

    // ---- typed declines ------------------------------------------------------

    #[test]
    fn declines_rather_than_guessing() {
        // Block-scoped sub-byte ELEMENT format (F4) — still outside the §6.1
        // closed set at sk4 → typed decline. (The two MX *scales* joined the
        // set at sk4 and now emit; see `sk4_mx_scales_go_from_silent_to_emitting`.)
        let bad_dtype = co(&[4096], DType::F4);
        assert_eq!(
            derive_structure_key_token(
                FuelOpCategory::BinaryElementwise,
                &[bad_dtype],
                "cuda:sm89"
            ),
            None
        );
        // A non-namespaced target is rejected (§6.8-0001 requires `<ns>:<cap>`).
        assert_eq!(
            derive_structure_key_token(FuelOpCategory::BinaryElementwise, &[f32c(&[4096])], "sm89"),
            None
        );
        // No operands → decline.
        assert_eq!(
            derive_structure_key_token(FuelOpCategory::BinaryElementwise, &[], "cuda:sm89"),
            None
        );
        // Over MAX_OPERANDS (8) → decline (§6.4-0002).
        let nine = vec![f32c(&[4096]); 9];
        assert_eq!(
            derive_structure_key_token(FuelOpCategory::BinaryElementwise, &nine, "cuda:sm89"),
            None
        );
        // A malformed descriptor (shape/strides length mismatch) → decline.
        let mut broken = f32c(&[4096]);
        broken.strides = vec![1, 1];
        assert_eq!(
            derive_structure_key_token(FuelOpCategory::BinaryElementwise, &[broken], "cuda:sm89"),
            None
        );
    }

    // ---- reduction cells: §6.5-0009(b) + the KISS A.1 shared vectors --------

    /// A reduced innermost axis derives v1 (§6.5-0009(b)) — the sk2-era
    /// deriver emitted v4 here, diverging from the (unchanged) spec clause.
    #[test]
    fn reduction_vec_width_is_v1_when_innermost_axis_reduced() {
        let token = derive_structure_key_token(
            FuelOpCategory::Reduction(ReduceAxes::All),
            &[f32c(&[4096])],
            "cuda:sm89",
        )
        .expect("reduction must derive");
        assert_eq!(
            token,
            "sk4|red|f32|cuda:sm89|ix32|grid|r1|co/00/v1/d16/f|rall"
        );
    }

    /// A keepdim mask that does NOT cover the innermost axis keeps the
    /// vectorized width (the v1 gate reads the innermost mask bit).
    #[test]
    fn reduction_v1_gate_reads_the_innermost_mask_bit() {
        // rank-2, reducing axis 0 only (mask 0x01): the innermost axis 1 is
        // NOT reduced → v4 stands.
        let token = derive_structure_key_token(
            FuelOpCategory::Reduction(ReduceAxes::Keepdim(0x01)),
            &[f32c(&[128, 256]), f32c(&[1, 256])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk4|red|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f|x01"
        );
    }

    /// KISS A.1: reduction keepdim `[4,8] → [4,1]` (trailing-axis ⇒ `rlast`).
    #[test]
    fn kiss_a1_reduction_trailing_axis_golden() {
        let token = derive_structure_key_token(
            FuelOpCategory::Reduction(ReduceAxes::TrailingAxis),
            &[f32c(&[4, 8]), f32c(&[4, 1])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk4|red|f32|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/00/v1/da/f|rlast"
        );
    }

    /// KISS A.1: reduction keepdim `[4,8] → [1,1]` (all-axes ⇒ `rall`).
    #[test]
    fn kiss_a1_reduction_all_axes_golden() {
        let token = derive_structure_key_token(
            FuelOpCategory::Reduction(ReduceAxes::All),
            &[f32c(&[4, 8]), f32c(&[1, 1])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk4|red|f32|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/00/v1/da/f|rall"
        );
    }

    /// KISS A.1: rank-1 reduction `[8] → [1]` — the §6.6-0009 tiebreak encodes
    /// `rall`, never `rlast`.
    #[test]
    fn kiss_a1_reduction_rank1_all_axes_golden() {
        let token = derive_structure_key_token(
            FuelOpCategory::Reduction(ReduceAxes::All),
            &[f32c(&[8]), f32c(&[1])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk4|red|f32|cuda:sm89|ix32|warp|r1|co/00/v1/d8/f;co/00/v1/da/f|rall"
        );
    }

    /// KISS A.1: rank-4 reduction over axes 1 and 3 ⇒ explicit keepdim
    /// bitmask `x0a`, work class `block`.
    #[test]
    fn kiss_a1_reduction_keepdim_mask_golden() {
        let token = derive_structure_key_token(
            FuelOpCategory::Reduction(ReduceAxes::Keepdim(0x0a)),
            &[f32c(&[2, 4, 3, 5]), f32c(&[2, 1, 3, 1])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk4|red|f32|cuda:sm89|ix32|block|r4|co/00/v1/da/f;co/00/v1/da/f|x0a"
        );
    }

    // ---- §6.5/§6.6 derivation pins (spec-conformance fixes) -----------------

    /// The innermost axis is axis rank−1 (§6.3-0011) even when its extent is
    /// 1: a `[4,1]` operand buckets `da` and derives v1 (the sk2-era
    /// rposition(extent>1) inner axis read extent 4 ⇒ d4/v4).
    #[test]
    fn trailing_unit_axis_reads_rank_minus_1_inner() {
        let token = derive_structure_key_token(
            FuelOpCategory::BinaryElementwise,
            &[f32c(&[4, 1])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(token, "sk4|bin|f32|cuda:sm89|ix32|warp|r2|co/00/v1/da/f|-");
    }

    /// A zero inner extent buckets `da` AND vectorizes `v1` — the coherent
    /// zero-extent pair. Both clauses carry the same `E >= N` guard: `div_bucket`
    /// (§6.5-0012, a guardless `0 % 16 == 0` would mis-bucket `d16`) and the
    /// vector-width ladder (§6.5-0009(c), a guardless `0 % L == 0` would
    /// mis-derive `v4` — the vacuous-truth trap KISS #82 F4 / PR #87 pinned to
    /// v1). This test previously froze the pre-fix `v4/da` — the incoherent pair
    /// where only one of the two axis clauses was guarded.
    #[test]
    fn zero_extent_buckets_da() {
        let token = derive_structure_key_token(
            FuelOpCategory::BinaryElementwise,
            &[f32c(&[0])],
            "cuda:sm89",
        )
        .expect("derives (never panics)");
        assert_eq!(token, "sk4|bin|f32|cuda:sm89|ix32|warp|r1|co/00/v1/da/f|-");
    }

    /// A fully reversed view is `co` under the |stride| layout algorithm
    /// (§6.5-0002) — the reversal lives only in the flipped flag (`r`).
    #[test]
    fn flipped_full_reverse_is_contiguous_per_abs_stride() {
        // shape [4,3], flip dim0: strides [-3,1], start_offset 9.
        let layout = Layout::new(
            Shape::from(vec![4usize, 3]),
            [-3isize, 1].into_iter().collect::<StrideVec>(),
            9,
        );
        let desc = FdxOperandDesc::from_layout(&layout, DType::F32);
        let token =
            derive_structure_key_token(FuelOpCategory::BinaryElementwise, &[desc], "cuda:sm89")
                .expect("derives");
        assert_eq!(token, "sk4|bin|f32|cuda:sm89|ix32|warp|r2|co/00/v1/da/r|-");
    }

    /// `alignment = 0` (unspecified base) cannot honor a packed load ⇒ v1
    /// (§6.5-0009; the sk2-era `0 % vbytes == 0` would have vectorized).
    #[test]
    fn alignment_zero_derives_v1() {
        let mut desc = f32c(&[4096]);
        desc.align_bytes = 0;
        let token =
            derive_structure_key_token(FuelOpCategory::BinaryElementwise, &[desc], "cuda:sm89")
                .expect("derives");
        assert_eq!(token, "sk4|bin|f32|cuda:sm89|ix32|grid|r1|co/00/v1/d16/f|-");
    }

    /// Work class and per-operand masks read the ITERATION FRAME
    /// (§6.5-0010 / §6.6-0013): a rank-deficient operand-0 is right-aligned,
    /// its missing frame axis broadcast (stride 0) — so the cell is `grid`
    /// (frame 128·256), not `block` (operand-0's own 256), and operand-0's
    /// sub-key is `br/01/v1/d16/f`.
    #[test]
    fn work_class_and_masks_use_the_iteration_frame() {
        let token = derive_structure_key_token(
            FuelOpCategory::BinaryElementwise,
            &[f32c(&[256]), f32c(&[128, 256]), f32c(&[128, 256])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk4|bin|f32|cuda:sm89|ix32|grid|r2|br/01/v1/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-"
        );
    }

    /// KISS A.1(b): an explicit stride-0 broadcast operand (same rank)
    /// derives `br`, mask 01, scalar width — byte-for-byte the staged
    /// `a1_elementwise_with_broadcast_operand` golden.
    #[test]
    fn kiss_a1_broadcast_operand_golden() {
        let bcast = FdxOperandDesc::from_layout(
            &Layout::new(
                Shape::from(vec![128usize, 256]),
                [0isize, 1].into_iter().collect::<StrideVec>(),
                0,
            ),
            DType::F32,
        );
        let token = derive_structure_key_token(
            FuelOpCategory::BinaryElementwise,
            &[f32c(&[128, 256]), bcast, f32c(&[128, 256])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk4|bin|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;br/01/v1/d16/f;co/00/v4/d16/f|-"
        );
    }

    /// KISS A.1: the canonical rank-2 binary cell (3 operands) and the
    /// in-place accumulate cell (2 operands — the read-modify-write operand
    /// appears exactly once, §6.6-0014) — byte-for-byte the staged
    /// `a1_elementwise_binary_canonical` / `a1_binary_two_operands` goldens.
    #[test]
    fn kiss_a1_bin_canonical_and_inplace_goldens() {
        let op = f32c(&[128, 256]);
        let canonical = derive_structure_key_token(
            FuelOpCategory::BinaryElementwise,
            &[op.clone(), op.clone(), op.clone()],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            canonical,
            "sk4|bin|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-"
        );
        let inplace = derive_structure_key_token(
            FuelOpCategory::BinaryElementwise,
            &[op.clone(), op],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            inplace,
            "sk4|bin|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f|-"
        );
    }

    /// KISS A.1: unary elementwise f16 `[64,128]` derives v8 (2-byte dtype) —
    /// byte-for-byte the staged `a1_unary_f16_v8` golden modulo op family
    /// (Fuel's category enum has no `une`; the operand derivation is shared,
    /// so the f16/v8 sub-keys are pinned via a `bin` cell).
    #[test]
    fn f16_wide_vector_matches_kiss_a1_subkeys() {
        let op = co(&[64, 128], DType::F16);
        let token = derive_structure_key_token(
            FuelOpCategory::BinaryElementwise,
            &[op.clone(), op],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk4|bin|f16|cuda:sm89|ix32|grid|r2|co/00/v8/d16/f;co/00/v8/d16/f|-"
        );
    }
}
