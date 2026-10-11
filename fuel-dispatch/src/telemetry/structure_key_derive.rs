// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fuel's INDEPENDENT `structure_key` derivation — the second implementation
//! for the two-implementation freeze-gate (KISS-CLASSIFY §6.6/§6.7).
//!
//! This is deliberately **Baracuda-free**: it recomputes the same `sk5` token
//! from Fuel's own [`FdxOperandDesc`] projection, with **no** `baracuda_kernels_*`
//! import, so a byte-match against Baracuda's emitted token is a genuine
//! two-implementation agreement. (K1 opacity — "Fuel never derives the key" —
//! governs the DISPATCH seam in [`super::structure_key`]; the freeze-gate is the
//! deliberate exception: Fuel derives the key independently *to check* it, never
//! to route.)
//!
//! Schema version: **sk5** (KISS-CLASSIFY, KISS `a33282c`; vectors as regenerated
//! by KISS #536 at `5b28bf51`; `contract_version` 3). Derived from the spec clauses, not from Baracuda's
//! implementation.
//!
//! # The sk4→sk5 delta this deriver implements
//!
//! One schema event (KISS #263 ruling D, #520, #517):
//! - **Field 1 is a derived `cell_class`, not a caller-chosen op family**
//!   (§6.5-0006, §6.5-0015). The 24 three-letter op-family codes are gone;
//!   the seven classes `ctr`, `ndx`, `rdc`, `ew1`, `ew2`, `ew3`, `oth` are
//!   **derived** from the operands and three caller-declared facts (the
//!   M/N/K roles, the index-operand slot, the reduced-axis set). No op name
//!   and no op family is an input — [`DeclaredFacts`] carries facts, never a
//!   category. A scan, a softmax, a normalization and a unary activation over
//!   one same-shape input and output therefore derive the **same** class
//!   (`ew1`); telling them apart is the op identity's job, which a lookup now
//!   carries beside the key (§6.6-0002).
//! - **`dequant_form` is keyed** (§6.3-0009a, §6.6-0007, §6.7-0004): an operand
//!   that carries a `quant` record gets an optional sixth sub-key component,
//!   `dl` (linear) or `dc` (codebook). An operand with no `quant` record is
//!   byte-identical to the five-component form.
//! - **An own axis of extent 1 counts as stride 0 for the broadcast mask**
//!   (§6.6-0008, #520), whatever stride the descriptor records for it.
//! - **The contraction field is present iff the class is `ctr`** and the reduce
//!   field is non-`-` iff the class is `rdc` (§6.6-0010, §6.6-0017). A declared
//!   reduced-axis set that is *empty* in the iteration frame is not a
//!   reduction, so it derives the class its shape gives it — never `rdc` with
//!   a `-` reduce field (`RdcWithoutReduce`).
//! - The version prefix is `sk5|`, and `f8e6m2` is **reserved** (§6.1-0013,
//!   KISS #517): see [`sk5_dtype_token`].
//!
//! # What this deliberately does NOT do
//!
//! - It is **emit-only**. It has no token *parse* path, so the new reader-side
//!   declines (`UnknownCellClass`, `RdcWithoutReduce`, `BadDequantForm`,
//!   `BadOperandSubKey`) have no site here; their emit-side dual is that the
//!   bad token is **unrepresentable** — [`CellClass`] and [`DequantForm`] are
//!   closed, wildcard-free enums.
//! - It does **not** change `fuel_ir`'s global dtype reservation. KISS made
//!   `f8e6m2` reserved, but Fuel has a real, implemented `DType::F8E6M2`, and
//!   `fuel_ir::RESERVED_DTYPE_TOKENS` also gates `DType::from_str` and FKC
//!   lowering. Reserving it there is a separate ruling; this module only
//!   refuses to *key* it (GAP-188 family; see `docs/gaps.md`).
//! - The production path (`baracuda_provider::map_op_category` → Baracuda's
//!   codec) is untouched and still derives `sk4` until Baracuda follows
//!   Unpopped's keyer. The two therefore **differ in version on purpose** for
//!   now; this deriver's conformance claim is the sk5 byte-match, not
//!   equality with production.
//!
//! # History
//!
//! The **sk3→sk4** respelling (dtype vocabulary, `sk4|` prefix) and the
//! earlier **sk2→sk3** delta (`gem` contraction precision coordinates, the
//! variant-explicit FP8 spellings, the `(acc + mp)` non-contraction field) are
//! all carried forward unchanged. The §6.5 algorithms were aligned with the
//! pinned text where the sk2-era code had latent divergences: the innermost
//! axis is axis `rank−1` (§6.3-0011), the divisibility ladder carries the
//! `E ≥ N` guard so `E = 0` buckets `da` (§6.5-0012), a reduction cell's
//! reduced innermost axis derives `v1` (§6.5-0009(b)), the layout tag follows
//! the 4-step `|stride|` algorithm (§6.5-0002), the work class reads the
//! iteration frame (§6.5-0010), and rank-deficient operands are right-aligned
//! into the frame (§6.6-0013).

use super::structure_key::FdxOperandDesc;
use fuel_ir::DType;

/// §6.4-0001 — the largest rank a cell operand may carry.
const MAX_RANK: usize = 8;
/// §6.4-0002 — the largest number of operands a cell may carry.
const MAX_OPERANDS: usize = 8;
/// §6.4-0004 — the largest serialized token length, in bytes.
const MAX_TOKEN_BYTES: usize = 4096;

/// The reduced-axis set of a reduction — the declared reduce fact (§6.6-0009 /
/// §6.7-0005). **Describes the INTENT, not the emitted token**: the actual
/// `structure_key` spelling (`-`, `rall`, `rlast`, or `x<hh>`) depends on
/// BOTH this axis set AND the cell's rank, by the rank-aware mask
/// canonicalization in `ReduceAxes::mask` / `reduce_field` — not on a
/// fixed per-variant spelling. For example `TrailingAxis` at rank 1 is
/// indistinguishable from "every axis reduced" and canonicalizes to `rall`,
/// not `rlast`; `All` at rank 0 has no axis to reduce at all, so the set is
/// *empty* and the cell is not a reduction (§6.5-0015 rule 3); and a `Keepdim`
/// mask that happens to equal the full-rank or trailing-bit mask
/// canonicalizes to `rall`/`rlast` just like the named variants would.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceAxes {
    /// Every axis reduced.
    All,
    /// Only the trailing (innermost) axis.
    TrailingAxis,
    /// An explicit keepdim bitmask for any other axis set.
    Keepdim(u8),
}

impl ReduceAxes {
    /// The effective reduced-axis bitmask at a given iteration-frame `rank`,
    /// independent of which variant produced it (§6.6-0009/§6.7-0005:
    /// `structure_key` canonicalizes by OPERATION, not by Rust variant). `0`
    /// for `rank == 0` (there is no axis to reduce). The SOLE source of this
    /// mask for both the class derivation (a non-empty set is `rdc`), the
    /// [`reduce_field`] token and the `v1` innermost-reduced gate — those
    /// derivations drifted once already (a rank-0 `All`/`TrailingAxis`
    /// reduction and an equivalent `Keepdim(0)` disagreed on whether the
    /// innermost axis was reduced), which is exactly the class of bug a single
    /// shared derivation forecloses structurally.
    fn mask(self, rank: usize) -> u8 {
        let full_mask: u8 = if rank == 0 {
            0
        } else if rank >= 8 {
            0xFF
        } else {
            (1u8 << rank) - 1
        };
        let trailing_bit: u8 = if rank == 0 { 0 } else { 1u8 << (rank - 1) };
        match self {
            ReduceAxes::All => full_mask,
            ReduceAxes::TrailingAxis => trailing_bit,
            ReduceAxes::Keepdim(m) => m,
        }
    }
}

/// The reduce field (§6.6-0009) for an effective reduced-axis `mask` at `rank`.
///
/// The caller gates this on the class being `rdc` (§6.6-0017: non-`-` iff
/// `rdc`); it is also total, so an empty mask spells `-` and a non-`rdc` cell
/// passes `0`. KISS-CLASSIFY §6.6-0009/§6.7-0005: a reduced set spanning every
/// axis is `rall` regardless of which constructor the caller used, and a
/// reduced set that is EXACTLY the trailing axis is `rlast` — both because
/// `structure_key` canonicalizes by OPERATION, not by which Rust variant
/// produced it: two callers describing the same reduction must get
/// byte-identical tokens.
fn reduce_field(mask: u8, rank: usize) -> String {
    let full_mask: u8 = if rank >= 8 { 0xFF } else { (1u8 << rank) - 1 };
    let trailing_bit: u8 = if rank == 0 { 0 } else { 1u8 << (rank - 1) };
    if mask == 0 {
        "-".to_string()
    } else if rank > 0 && mask == full_mask {
        "rall".to_string()
    } else if rank > 0 && mask == trailing_bit {
        "rlast".to_string()
    } else {
        format!("x{mask:02x}")
    }
}

/// The math-precision key coordinate of a `ctr` cell — `<mp>` in the
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

/// The caller-declared role hints of a dense-contraction cell
/// (§6.6-0012/-0016): the M/N/K axis-role extents (an implementation MUST NOT
/// infer M/N/K from bare operand extents), the conditionally-present batch
/// extent, and the precision coordinates — weight / accumulator / output
/// dtypes plus the math-precision class (§6.7-0006).
///
/// **Declaring a `GemCell` is declaring the contraction roles**, which is what
/// makes the cell class `ctr` (§6.5-0015 rule 1): the class is derived from
/// this fact, never supplied beside it.
///
/// The dtype coordinates are Fuel [`DType`]s, not spellings: the closed §6.1
/// token set is applied by [`sk5_dtype_token`] at emission, so a reserved
/// (`fnuz`, `f8e6m2`) or out-of-set spelling is **unrepresentable** here —
/// build-time closure instead of a parse-time decline.
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

/// The **non-contraction** precision coordinate `(acc + mp)` — the optional
/// trailing field of a non-`ctr` cell (KISS-CLASSIFY §6.7-0013, realizing the
/// §6.7-0012 forward requirement; unchanged at sk5).
///
/// Spelled **gem-symmetrically** as `<acc>/<mp>`, and it occupies the *same*
/// optional-trailing slot the contraction group occupies for `ctr`. A cell
/// carries **at most one** precision field and the two **never coexist**, so
/// the 9-vs-10 field count resolves on the cell class alone.
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
    /// Math-precision coordinate `<mp>`, the same codes as `ctr`'s.
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
            sk5_dtype_token(self.acc_dtype)?,
            self.math_precision.code(),
        ))
    }
}

/// The closed, **derived** cell class a `structure_key` keys on (KISS-CLASSIFY
/// §6.5-0006, §6.5-0015) — a fact about the cell's structure, never an op
/// family and never supplied as a label.
///
/// The enum is the emit-side form of the reader's `UnknownCellClass` decline:
/// there is no eighth value to emit, and [`CellClass::code`] is wildcard-free,
/// so a new class is a compile error that forces a decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CellClass {
    /// `ctr` — dense contraction: the M/N/K axis roles are declared.
    DenseContraction,
    /// `ndx` — indexed access: an index-operand role is declared.
    IndexedAccess,
    /// `rdc` — reduction: a non-empty reduced-axis set is declared.
    Reduction,
    /// `ew1` — pointwise by shape, exactly one input operand.
    ElementwiseOne,
    /// `ew2` — pointwise by shape, exactly two input operands.
    ElementwiseTwo,
    /// `ew3` — pointwise by shape, exactly three input operands.
    ElementwiseThree,
    /// `oth` — none of the above (includes a pointwise cell with no input
    /// operand and one with more than three).
    Other,
}

impl CellClass {
    /// The §6.5-0006 token code.
    pub fn code(self) -> &'static str {
        match self {
            CellClass::DenseContraction => "ctr",
            CellClass::IndexedAccess => "ndx",
            CellClass::Reduction => "rdc",
            CellClass::ElementwiseOne => "ew1",
            CellClass::ElementwiseTwo => "ew2",
            CellClass::ElementwiseThree => "ew3",
            CellClass::Other => "oth",
        }
    }
}

/// The closed `dequant_form` set (§6.3-0009a): `dl` for `linear`, `dc` for
/// `codebook`. An operand that carries no `quant` record has no such component.
///
/// ⚠️ Not inferable from a quantization *family*, which Classify does not
/// interpret — the caller supplies it (§6.3-0009a). The one quant distinction
/// that is not a performance choice: no scale reproduces a codebook with a
/// linear dequant, so a shared key would return a numerically WRONG result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DequantForm {
    /// `dl` — recovered by an affine map of the stored integer and its scale.
    Linear,
    /// `dc` — recovered by indexing a table of values the stored bits select.
    Codebook,
}

impl DequantForm {
    fn code(self) -> &'static str {
        match self {
            DequantForm::Linear => "dl",
            DequantForm::Codebook => "dc",
        }
    }
}

/// An operand's role in the canonical order (§6.6-0014).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandRole {
    /// Read only.
    Input,
    /// Read **and** written (an accumulator, a read-modify-write operand).
    /// Appears exactly once, classified as an input at its position in the
    /// input call order (§6.6-0014), and also counts as a *written* operand
    /// for the pointwise test (§6.5-0015).
    InPlace,
    /// Written only.
    Output,
}

/// One operand of a cell: the FDX descriptor plus the two facts the sk5
/// derivation reads beyond it — its role (§6.6-0014) and, iff it carries a
/// `quant` record, its `dequant_form` (§6.3-0009a).
///
/// Kept OUTSIDE [`FdxOperandDesc`] on purpose: that type is the
/// `#[non_exhaustive]` wire descriptor shared with the production provider
/// (GAP-178), and this deriver's extra facts are a conformance concern.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CellOperand {
    /// The operand's layout/dtype descriptor.
    pub desc: FdxOperandDesc,
    /// The operand's role in the canonical order.
    pub role: OperandRole,
    /// `Some` iff the operand carries a `quant` record.
    pub dequant_form: Option<DequantForm>,
}

impl CellOperand {
    fn new(desc: FdxOperandDesc, role: OperandRole) -> Self {
        Self {
            desc,
            role,
            dequant_form: None,
        }
    }

    /// A read-only operand.
    pub fn input(desc: FdxOperandDesc) -> Self {
        Self::new(desc, OperandRole::Input)
    }

    /// A read-and-written operand.
    pub fn in_place(desc: FdxOperandDesc) -> Self {
        Self::new(desc, OperandRole::InPlace)
    }

    /// A written operand.
    pub fn output(desc: FdxOperandDesc) -> Self {
        Self::new(desc, OperandRole::Output)
    }

    /// Mark the operand as carrying a `quant` record with this `dequant_form`.
    #[must_use]
    pub fn with_dequant_form(mut self, form: DequantForm) -> Self {
        self.dequant_form = Some(form);
        self
    }
}

/// The three **caller-declared facts of the invocation** the cell class is
/// derived from (§6.5-0015, §6.6-0012). They are facts the op itself states
/// (KISS-Ops §6.13-0009 M/N/K iteration space, §6.11-0012 index operand role,
/// §6.19 `reduce_axes`), not categories, and **no op name or family appears
/// here** — the type has no way to say "this is a softmax".
///
/// `#[non_exhaustive]` with constructors, as [`FdxOperandDesc`] is (GAP-178):
/// a fourth declared fact would be a schema event, and should not also be a
/// source break for every constructing crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct DeclaredFacts {
    /// The M/N/K axis roles (and the contraction precision coordinates).
    /// `Some` ⇒ class `ctr` (rule 1).
    pub contraction: Option<GemCell>,
    /// The operand slot (an index into the canonical operand list) that
    /// carries the index/address role. `Some` ⇒ class `ndx` unless `ctr`
    /// (rule 2). It must name a non-output operand.
    pub index_operand: Option<usize>,
    /// The reduced-axis set. Non-empty in the iteration frame ⇒ class `rdc`
    /// unless `ctr`/`ndx` (rule 3).
    pub reduced_axes: Option<ReduceAxes>,
}

impl DeclaredFacts {
    /// No declared fact: the class is decided by the pointwise rules alone.
    pub fn none() -> Self {
        Self::default()
    }

    /// The contraction roles are declared.
    pub fn contraction(cell: GemCell) -> Self {
        Self {
            contraction: Some(cell),
            ..Self::default()
        }
    }

    /// The index-operand slot is declared.
    pub fn indexed(slot: usize) -> Self {
        Self {
            index_operand: Some(slot),
            ..Self::default()
        }
    }

    /// A reduced-axis set is declared.
    pub fn reduction(axes: ReduceAxes) -> Self {
        Self {
            reduced_axes: Some(axes),
            ..Self::default()
        }
    }
}

/// Is `target` a well-formed namespaced target (§6.8-0001/-0005): exactly one
/// `:` separating two non-empty parts, with neither part (nor the separator
/// itself) containing `|`, `;`, `/`, whitespace, or a control character.
/// `target.contains(':')` alone — the prior check — accepts `"a:b:c"`,
/// `":x"`, `"x:"`, and `"cuda:sm|89"`, all of which this rejects.
fn is_valid_namespaced_target(target: &str) -> bool {
    let mut parts = target.split(':');
    let (Some(ns), Some(rest)) = (parts.next(), parts.next()) else {
        return false;
    };
    if parts.next().is_some() {
        return false; // more than one ':'
    }
    if ns.is_empty() || rest.is_empty() {
        return false;
    }
    !target
        .chars()
        .any(|c| c == '|' || c == ';' || c == '/' || c.is_whitespace() || c.is_control())
}

/// The iteration frame (§6.6-0013): rank = widest operand rank (§6.6-0006);
/// frame extent per axis = the maximum extent across the right-aligned
/// operands at that axis. `None` for a cell with no operand, or one whose
/// operand is malformed or over the rank/operand caps.
fn iteration_frame(operands: &[CellOperand]) -> Option<Vec<i64>> {
    if operands.is_empty() || operands.len() > MAX_OPERANDS {
        return None; // MAX_OPERANDS cap (§6.4-0002)
    }
    if operands
        .iter()
        .any(|o| o.desc.shape.len() > MAX_RANK || o.desc.shape.len() != o.desc.strides.len())
    {
        return None; // MAX_RANK cap (§6.4-0001) / malformed descriptor
    }
    let rank = operands
        .iter()
        .map(|o| o.desc.shape.len())
        .max()
        .unwrap_or(0);
    let mut frame = vec![0i64; rank];
    for o in operands {
        let off = rank - o.desc.shape.len();
        for (i, &e) in o.desc.shape.iter().enumerate() {
            frame[off + i] = frame[off + i].max(e);
        }
    }
    Some(frame)
}

/// Derive the §6.5-0015 cell class: **first rule that holds decides.**
///
/// 1. the contraction roles are declared → `ctr`;
/// 2. else the index role is declared → `ndx`;
/// 3. else the reduced-axis set is non-empty (in the iteration frame) → `rdc`;
/// 4. else, if the cell is pointwise and has `n` input operands, `1 ≤ n ≤ 3`
///    → `ew<n>`;
/// 5. else `oth` — which includes a pointwise cell with no input operand and
///    one with more than three.
///
/// Reads only descriptor facts and the three declared facts: never an op name
/// or family, and no op-to-class table exists to read one from.
///
/// Returns `None` — a typed decline, never a wrong class — for a cell that is
/// not in canonical order (an input after an output, §6.6-0014), that names an
/// index slot which is absent or an output, or that is empty / over the
/// rank and operand caps. §6.5-0006 makes the class *total over valid cells*,
/// so the decline is about an ill-formed **cell**, never "no class fits".
pub fn derive_cell_class(facts: &DeclaredFacts, operands: &[CellOperand]) -> Option<CellClass> {
    let frame = iteration_frame(operands)?;
    cell_class_in_frame(facts, operands, &frame)
}

/// §6.6-0014 canonical order: every input / in-place operand precedes every
/// output. A violating cell is ill-formed, not "other".
fn is_canonical_order(operands: &[CellOperand]) -> bool {
    match operands.iter().position(|o| o.role == OperandRole::Output) {
        Some(p) => operands[p..].iter().all(|o| o.role == OperandRole::Output),
        None => true,
    }
}

/// §6.5-0015 rule 4's pointwise predicate: at every iteration-frame axis every
/// operand's extent there (its own, or 1 at an axis it does not have) is 1 or
/// the frame extent, and every WRITTEN operand (an output, or an in-place
/// operand) has exactly the frame extent at every axis and the iteration rank.
fn is_pointwise(operands: &[CellOperand], frame: &[i64]) -> bool {
    let rank = frame.len();
    operands.iter().all(|o| {
        let off = rank - o.desc.shape.len();
        let extents_ok = frame.iter().enumerate().all(|(i, &f)| {
            let e = if i < off { 1 } else { o.desc.shape[i - off] };
            e == 1 || e == f
        });
        let written = matches!(o.role, OperandRole::Output | OperandRole::InPlace);
        let written_ok = !written || o.desc.shape == frame;
        extents_ok && written_ok
    })
}

fn cell_class_in_frame(
    facts: &DeclaredFacts,
    operands: &[CellOperand],
    frame: &[i64],
) -> Option<CellClass> {
    if !is_canonical_order(operands) {
        return None;
    }
    if let Some(slot) = facts.index_operand {
        // The index role names an operand that is read; an out-of-range slot or
        // an output is not a role hint, it is an error.
        match operands.get(slot) {
            Some(o) if o.role != OperandRole::Output => {}
            _ => return None,
        }
    }

    if facts.contraction.is_some() {
        return Some(CellClass::DenseContraction);
    }
    if facts.index_operand.is_some() {
        return Some(CellClass::IndexedAccess);
    }
    if let Some(axes) = facts.reduced_axes
        && axes.mask(frame.len()) != 0
    {
        return Some(CellClass::Reduction);
    }

    let n_inputs = operands
        .iter()
        .filter(|o| o.role != OperandRole::Output)
        .count();
    Some(match (is_pointwise(operands, frame), n_inputs) {
        (true, 1) => CellClass::ElementwiseOne,
        (true, 2) => CellClass::ElementwiseTwo,
        (true, 3) => CellClass::ElementwiseThree,
        _ => CellClass::Other,
    })
}

/// Derive the KISS `sk5` `structure_key` token for a cell, independently of
/// Baracuda. `operands` are in canonical order — inputs (and in-place
/// operands) in call order, then outputs (§6.6-0014). Returns `None` (a typed
/// decline, never a wrong token) on an unmappable or reserved dtype, an empty
/// operand list, a rank over `MAX_RANK` (8), more than `MAX_OPERANDS` (8)
/// operands, a malformed descriptor, a cell not in canonical order, a
/// non-namespaced target, an invalid index slot, or an invalid (negative)
/// contraction role extent.
///
/// `acc_mp` is the non-contraction precision coordinate (§6.7-0013). Passing
/// `None` is **not** "unknown" — it asserts the §6.17-0005 diagonal
/// (accumulator == compute dtype, `<mp>` at default), which is exactly what an
/// absent field means. Supplying one for a `ctr` cell declines: §6.7-0013 pins
/// that a cell carries **at most one** precision field and that the two never
/// coexist, so emitting both would be invalid rather than merely odd.
pub fn derive_structure_key_token(
    facts: &DeclaredFacts,
    operands: &[CellOperand],
    target: &str,
    acc_mp: Option<AccMp>,
) -> Option<String> {
    if acc_mp.is_some() && facts.contraction.is_some() {
        return None;
    }
    let frame = iteration_frame(operands)?;
    let first = operands.first()?;
    // KISS-CLASSIFY §3.2: F8E8M0 is a ratified MX **scale**, valid only as a
    // sibling operand, never as the dense/primary dtype this generic path
    // reads from operand 0. `sk5_dtype_token` is the position-agnostic "is this
    // dtype spellable at all" table, shared with call sites where F8E8M0 IS
    // valid (e.g. `contraction_field`'s `<wdt>` field, which reads a
    // caller-supplied role hint, never operand 0 positionally). This decline is
    // scoped to `first.desc.dtype` specifically, not to the dtype itself, so it
    // does not narrow anything KISS ratifies. (`F8E6M2` is no longer here: it
    // is reserved outright at sk5 and `sk5_dtype_token` declines it everywhere.)
    if matches!(first.desc.dtype, DType::F8E8M0) {
        return None;
    }
    let dtype = sk5_dtype_token(first.desc.dtype)?;
    if !is_valid_namespaced_target(target) {
        return None; // namespaced target required (§6.8-0001/-0005)
    }

    let class = cell_class_in_frame(facts, operands, &frame)?;
    let rank = frame.len();

    // Field 4 — index width: max touched offset Σ|stride|·(ext−1) across
    // operands' own axes (§6.5-0011; a padded frame axis is stride-0 and
    // contributes 0).
    let max_touched: i128 = operands
        .iter()
        .map(|o| {
            o.desc
                .strides
                .iter()
                .zip(o.desc.shape.iter())
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

    // The reduced-axis mask is read ONLY for an `rdc` cell (§6.6-0017: the
    // reduce field is non-`-` iff `rdc`). A cell that declares a reduced set
    // but derives `ctr`/`ndx` keeps `-` and no `v1` reduction gate.
    //
    // §6.5-0009(b): every operand of a reduction cell whose reduced set
    // includes the innermost iteration-frame axis derives v1. Right-alignment
    // (§6.6-0013) maps every operand's innermost axis to frame axis rank−1,
    // so the gate is cell-level. Derived from the SAME rank-aware mask the
    // class derivation and the reduce field use.
    let reduced_mask = match (class, facts.reduced_axes) {
        (CellClass::Reduction, Some(axes)) => axes.mask(rank),
        _ => 0,
    };
    let innermost_reduced = rank >= 1 && (reduced_mask >> (rank - 1)) & 1 == 1;

    // Field 7 — per-operand sub-keys, canonical order, each derived in the
    // iteration frame.
    let operand_keys: Vec<String> = operands
        .iter()
        .map(|o| operand_sub_key(o, &frame, innermost_reduced))
        .collect();

    // Contraction field — present iff `ctr` (§6.6-0010).
    let contraction = match (class, facts.contraction) {
        (CellClass::DenseContraction, Some(cell)) => Some(contraction_field(&cell)?),
        _ => None,
    };

    let mut token = format!(
        "sk5|{class}|{dtype}|{target}|{idx}|{work}|r{rank}|{ops}|{reduce}",
        class = class.code(),
        idx = index_width,
        work = work_class,
        ops = operand_keys.join(";"),
        reduce = reduce_field(reduced_mask, rank),
    );
    // §6.7-0013(c)/(e): the non-contraction precision field is
    // OMITTED-WHEN-ABSENT — not `-`, not empty. This is a deliberate contrast
    // with the MANDATORY reduce field above, which emits `-` when
    // inapplicable. Applying the reduce field's convention here would emit a
    // spurious trailing `|-` and fail the byte-match. The nearest precedent in
    // this very token teaches the opposite rule, which is what makes it a trap.
    if let Some(a) = acc_mp.filter(|a| a.deviates_from(first.desc.dtype)) {
        token.push('|');
        token.push_str(&a.field()?);
    }
    if let Some(c) = contraction {
        token.push('|');
        token.push_str(&c);
    }
    if token.len() > MAX_TOKEN_BYTES {
        return None; // §6.4-0004 — declines, never truncates
    }
    Some(token)
}

/// The `ctr` contraction field (§6.7-0006):
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
    let wdt = sk5_dtype_token(cell.weight_dtype)?;
    let acc = sk5_dtype_token(cell.acc_dtype)?;
    let out = sk5_dtype_token(cell.out_dtype)?;
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

/// KISS-CLASSIFY §6.1 dtype token for a keyed dtype coordinate — the §6.1
/// vocabulary introduced at sk4 and **unchanged at sk5**, minus what sk5
/// reserves.
///
/// FP8 tokens are width-prefixed and variant-explicit: Fuel's `F8E4M3` is the
/// OCP format → `f8e4m3fn`, and `F8E5M2` → `f8e5m2` **unsuffixed**, since only
/// the `fnuz` layouts deviate from IEEE E5M2. The reserved `fnuz` variants have
/// no Fuel `DType` and therefore can never be emitted here.
///
/// # `F8E6M2` is RESERVED at sk5 and is declined here — and only here
///
/// KISS #517 (§6.1-0013; `kiss-classify-vocab` 0.4.0 `Dtype::is_reserved`,
/// agreeing with KISS's `dtype_manifest.json` `reserved` column) made
/// `f8e6m2` a **reserved** spelling: recognized on parse, no encoding pinned,
/// no op assigns it computation semantics, so any use in a keyed position is a
/// typed decline distinct from an unknown token. Fuel's `DType::F8E6M2` is a
/// real, implemented dtype, so this wrapper — not
/// [`fuel_ir::sk4_token`] and not `fuel_ir::RESERVED_DTYPE_TOKENS` — is where
/// the sk5 reservation takes effect for **keying**. Reserving it in `fuel_ir`
/// would also break `DType::from_str("f8e6m2")` and FKC lowering, which is a
/// separate ruling (see `docs/gaps.md`).
///
/// **Declining here aborts the WHOLE derivation** (`?` at the call site), not
/// just this field — so a dtype moving between the emitted and declined sets
/// changes whether a cell emits a structure key at all.
///
/// # §6.1-0001 has no site in Fuel at all — emitter OR parser
///
/// That clause binds a **reader of a `structure_key`**, and Fuel is not one: it
/// derives and emits keys, and treats foreign ones as opaque bytes (K1
/// opacity). Fuel's KISS-Classify conformance claim is scoped to the **emit**
/// surface. The reserved-token declines that do exist —
/// `fuel_ir::RESERVED_DTYPE_TOKENS` and `FkcError::ReservedScalarType` on
/// hand-authored `.fkc.md` input — are diagnostics on Fuel's own ingest
/// surfaces, NOT §6.1-0001 conformance.
///
/// The positional restriction — an MX scale is valid only as a sibling operand,
/// never as the generic path's dense/primary dtype (KISS-CLASSIFY §3.2) — is
/// NOT enforced here and cannot be: this function is the position-agnostic "is
/// this dtype spellable at all" table. The enforcement lives at
/// [`derive_structure_key_token`]'s call site, which declines before ever
/// calling this function if operand 0 is `F8E8M0`.
///
/// The block-scoped sub-byte **element** formats (`F6E2M3`/`F6E3M2`/`F4`) are
/// still outside the set and typed-decline (`None`), never a guessed token.
///
/// **Delegates to [`fuel_ir::sk4_token`], the single source of the seam
/// spelling**; this function used to carry its own copy of the table. Two
/// hand-maintained copies of one mapping is the drift pattern this repo keeps
/// re-finding: the copy that stops being edited does not fail loudly, it emits
/// a **retired spelling under a current version prefix** — which §6.1-0004
/// forbids outright, and which no test downstream of this function can detect,
/// because the key it produces is still well-formed. `sk4_token` is a
/// wildcard-free `match`, so a new Fuel `DType` is still a compile error
/// forcing a decision in the crate that owns `DType`.
pub fn sk5_dtype_token(dt: DType) -> Option<&'static str> {
    match dt {
        DType::F8E6M2 => None,
        other => fuel_ir::sk4_token(other),
    }
}

/// One operand's `<contig>/<bcasthex>/<vec>/<div>/<flip>[/<dequant>]` sub-key
/// (§6.6-0007), derived in the iteration frame (§6.6-0013): a rank-deficient
/// operand is right-aligned, with every frame axis below `rank − r` treated as
/// broadcast (stride 0) for it.
///
/// KISS-CLASSIFY §6.5-0014, scope C (KISS#519, PM-ruled 2026-10-02): the
/// broadcast-axis MASK still reads the frame-padded view (§6.6-0008/
/// §6.6-0013) -- a padded axis contributes to admissibility regardless of
/// the operand's own rank. But `layout_tag`, vector width, and the
/// divisibility bucket read the operand's OWN axes only. Padding a
/// rank-deficient operand as broadcast used to make `layout_tag` `br`
/// REGARDLESS of the operand's own strides, so a dense `[256]` operand in a
/// `[128,256]` frame collided with a genuinely strided/gathered rank-1
/// operand under one sub-key -- two operands with different real memory
/// behavior, one spelling. Splitting the two concerns (admissibility mask
/// vs. this operand's own access pattern) is what KISS#519 fixes.
fn operand_sub_key(co: &CellOperand, frame: &[i64], innermost_reduced: bool) -> String {
    let o = &co.desc;
    let rank = frame.len();
    let off = rank - o.shape.len();

    // Broadcast-axis mask (§6.6-0008): bit i set iff iteration-frame axis i
    // has extent > 1 and this operand's stride along it is 0 -- computed
    // over the frame-PADDED view, since a padded (rank-deficient) axis is a
    // broadcast axis by construction regardless of what layout_tag reports.
    //
    // sk5 (#520): an OWN axis of extent 1 also counts as stride 0, whatever
    // stride the descriptor records -- the stride of a unit axis is never
    // stepped and is arbitrary, so the bit must not depend on it. At sk4 an
    // extent-1 axis with a non-zero recorded stride left its bit clear and
    // the same tensor spelled two tokens.
    let mut mask = 0u8;
    for (i, &f) in frame.iter().enumerate().take(rank.min(8)) {
        let stride_is_zero = if i < off {
            true
        } else {
            o.strides[i - off] == 0 || o.shape[i - off] == 1
        };
        if f > 1 && stride_is_zero {
            mask |= 1 << i;
        }
    }

    // layout_tag reads the operand's OWN extents/strides only (§6.5-0014
    // scope C) -- NOT the padded view. A rank-deficient operand's missing
    // axes are an admissibility fact (captured in `mask` above), not a
    // statement about how THIS operand's own memory is laid out.
    let layout = layout_code(&o.shape, &o.strides);

    // Own innermost axis (§6.3-0011): axis rank−1 of the operand's OWN shape
    // (right-aligned to the frame innermost). A rank-0 operand has none.
    let (inner_extent, inner_stride) = match o.shape.len().checked_sub(1) {
        Some(i) => (o.shape[i], Some(o.strides[i])),
        None => (1, None),
    };

    let div = div_bucket(inner_extent);

    // Vector-access width (§6.5-0009 / §6.5-0013): reads the operand's OWN
    // axes (§6.5-0014 scope C) -- `mask != 0` (a FRAME-padded broadcast) no
    // longer forces v1 on its own; only the operand's OWN layout being `br`
    // does. Still v1 on: own layout `br`, a reduced innermost axis of an
    // `rdc` cell, a missing/sub-byte/unaligned base, or a non-forward-unit
    // inner stride; else the largest L ∈ {8,4,2} within the 16-byte cap
    // whose exact modulo divides the alignment and the inner extent.
    let dsz = o.dtype.size_in_bytes();
    let vec = if layout == "br"
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

    let mut key = format!(
        "{layout}/{mask:02x}/{vec}/{div}/{flip}",
        flip = if o.flipped { "r" } else { "f" },
    );
    // §6.6-0007 / §6.3-0009a: the sixth component, iff the operand carries a
    // `quant` record. An operand with none stays byte-identical to the
    // five-component form.
    if let Some(form) = co.dequant_form {
        key.push('/');
        key.push_str(form.code());
    }
    key
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

    /// Canonical order for the sk4-era construction idiom `(inputs…, output)`:
    /// the LAST descriptor is the output and the rest are inputs; a lone
    /// descriptor is an input. In-place (read-modify-write) cells and cells
    /// with a different role layout are built explicitly with
    /// [`CellOperand::in_place`] / [`CellOperand::output`].
    fn canon(descs: &[FdxOperandDesc]) -> Vec<CellOperand> {
        let n = descs.len();
        descs
            .iter()
            .enumerate()
            .map(|(i, d)| {
                if n >= 2 && i == n - 1 {
                    CellOperand::output(d.clone())
                } else {
                    CellOperand::input(d.clone())
                }
            })
            .collect()
    }

    fn derive(facts: DeclaredFacts, descs: &[FdxOperandDesc], target: &str) -> Option<String> {
        derive_structure_key_token(&facts, &canon(descs), target, None)
    }

    fn derive_acc_mp(
        facts: DeclaredFacts,
        descs: &[FdxOperandDesc],
        target: &str,
        acc_mp: Option<AccMp>,
    ) -> Option<String> {
        derive_structure_key_token(&facts, &canon(descs), target, acc_mp)
    }

    /// **SPELLED and DERIVABLE are different axes, and only one of them was ever
    /// measured.** [`sk5_dtype_token`] maps a `DType` to its seam token — that is
    /// *spelling*, and `token_kind.rs` already asserts a partition over the
    /// underlying vocabulary. This asks the other question: **can Fuel construct
    /// a real cell whose operand 0 carries that dtype, so a token containing the
    /// spelling actually comes out?**
    ///
    /// Written because a cross-project record carried three dtypes (`i16`,
    /// `f8e8m0`, `f8e6m2`) as *"legal token but not derivable"* — a Fuel-side
    /// capability gap. The claim conflated the axes. Rather than answer for those
    /// three and leave the rest assumed, this measures **every** dtype that has a
    /// spelling, so the answer cannot rot into a claim about a subset someone
    /// happened to ask about.
    ///
    /// The assertion is `spelled ⇒ derivable`, and it is deliberately exhaustive
    /// over `DType::ALL`: a future dtype that can be spelled but not built at
    /// operand 0 fails **here**, with its name, instead of being discovered by a
    /// peer.
    ///
    /// **One NAMED exemption, not a weakening of the invariant: `F8E8M0`.** It is
    /// spelled but DELIBERATELY not derivable at operand 0 — KISS-CLASSIFY §3.2
    /// ratifies it as an MX scale, valid only as a sibling operand, and
    /// [`derive_structure_key_token`] declines it there on purpose. The
    /// assertion checks the exemption set is EXACTLY that one dtype, so a future
    /// dtype landing in `not_derivable` unexpectedly still fails loudly.
    ///
    /// **`F8E6M2` is no longer in this population at all (sk5, KISS #517):** it
    /// is reserved, so it has no spelling here. It is asserted separately, below
    /// and in `f8e6m2_is_reserved_at_sk5_in_every_keyed_position`.
    #[test]
    fn every_spelled_dtype_is_also_derivable_at_operand_zero() {
        let mut spelled = 0usize;
        let mut not_derivable: Vec<(DType, &'static str)> = Vec::new();
        let mut mismatched: Vec<(DType, &'static str, String)> = Vec::new();

        for dt in DType::ALL.iter().copied() {
            let Some(spelling) = sk5_dtype_token(dt) else {
                continue; // no sk5 seam token — outside this axis by construction
            };
            spelled += 1;

            // The simplest cell that puts a dtype at operand 0.
            let ops = [co(&[4096], dt), co(&[4096], dt)];
            match derive(DeclaredFacts::none(), &ops, "cuda:sm89") {
                None => not_derivable.push((dt, spelling)),
                Some(token) => {
                    // Field 3 (0-indexed 2) is `<dtype>`: sk5|<class>|<dtype>|<target>|…
                    let got = token.split('|').nth(2).unwrap_or("<missing>");
                    if got != spelling {
                        mismatched.push((dt, spelling, got.to_string()));
                    }
                }
            }
        }

        // Non-vacuity: if the spelling table ever returned `None` for everything,
        // the loop above would pass by examining nothing. 14 = the 15 spelled at
        // sk4 minus `f8e6m2`, reserved at sk5; `token_kind.rs` pins the 15
        // independently.
        assert_eq!(
            spelled, 14,
            "expected 14 sk5-spelled dtypes; got {spelled} — this test would be vacuous at 0",
        );
        let known_positional_exemptions = [DType::F8E8M0];
        let unexpected_gaps: Vec<(DType, &'static str)> = not_derivable
            .iter()
            .copied()
            .filter(|(dt, _)| !known_positional_exemptions.contains(dt))
            .collect();
        assert!(
            unexpected_gaps.is_empty(),
            "spelled but NOT derivable at operand 0, and NOT a known positional \
             exemption: {unexpected_gaps:?} — a dtype with a seam token that no \
             cell can carry is a real capability gap",
        );
        let exempted_but_derivable: Vec<DType> = known_positional_exemptions
            .iter()
            .copied()
            .filter(|dt| !not_derivable.iter().any(|(nd, _)| nd == dt))
            .collect();
        assert!(
            exempted_but_derivable.is_empty(),
            "{exempted_but_derivable:?} are expected to DECLINE at operand 0 \
             (KISS-CLASSIFY §3.2 positional restriction) but derived a token — \
             the positional decline regressed",
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

    // ---- (a) sk5 prefix on every token class --------------------------------

    /// The relu_add f32 grid-stride freeze-gate cell (condition-1): 3 rank-1
    /// f32 operands [4096], contiguous, offset 0 (align 256): in0, in1, out.
    /// Byte-for-byte the KISS PR #81 staged golden
    /// (`relu_add_generated_r1_cell`).
    #[test]
    fn fuel_derives_relu_add_sk5_token_byte_for_byte() {
        let op = f32c(&[4096]);
        let token = derive(
            DeclaredFacts::none(),
            &[op.clone(), op.clone(), op],
            "cuda:sm89",
        )
        .expect("relu_add f32 must derive a token");
        assert_eq!(
            token,
            "sk5|ew2|f32|cuda:sm89|ix32|grid|r1|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-"
        );
    }

    /// Every derivable token carries the sk5 prefix and no retired bytes — and
    /// none carries a retired sk4 op-family code in field 2: the field is one
    /// of the seven derived class codes, whatever facts were declared.
    #[test]
    fn sk5_prefix_and_a_derived_class_code_on_every_token_class() {
        const CLASS_CODES: [&str; 7] = ["ctr", "ndx", "rdc", "ew1", "ew2", "ew3", "oth"];
        const RETIRED_FAMILY_CODES: [&str; 19] = [
            "une", "bin", "ter", "scn", "red", "gem", "nrm", "cnv", "pol", "idx", "shp", "srt",
            "fft", "lin", "rnd", "seg", "sft", "att", "los",
        ];
        let op = f32c(&[4096]);
        let facts = [
            DeclaredFacts::none(),
            DeclaredFacts::reduction(ReduceAxes::All),
            DeclaredFacts::reduction(ReduceAxes::TrailingAxis),
            DeclaredFacts::reduction(ReduceAxes::Keepdim(0x01)),
            DeclaredFacts::contraction(gem_f32(8, 4096, 4096)),
            DeclaredFacts::indexed(0),
        ];
        for f in facts {
            let token = derive(f, std::slice::from_ref(&op), "cuda:sm89")
                .unwrap_or_else(|| panic!("{f:?} must derive"));
            assert!(token.starts_with("sk5|"), "{token} lacks the sk5 prefix");
            assert!(!token.contains("sk2") && !token.contains("sk4"), "{token}");
            let class = token.split('|').nth(1).unwrap();
            assert!(
                CLASS_CODES.contains(&class),
                "{token}: `{class}` not a class"
            );
            assert!(
                !RETIRED_FAMILY_CODES.contains(&class),
                "{token}: retired op-family code `{class}` in field 2"
            );
        }
    }

    // ---- (e) non-gem tokens: the sk2-era battery, field by field ------------

    /// The four committed sk2-era battery cells (pinned inline from the
    /// fdc1e987/97307020 test battery). At sk2 the first two fields were
    /// `sk2|<op family>`; at sk5 they are `sk5|<derived cell class>`. Everything
    /// from field 4 on — target, index width, work class, rank, operand
    /// sub-keys, reduce field — must be **byte-identical** to the sk2 token,
    /// and the dtype field is identical except the `s16` -> `i16` respell.
    /// That is the continuity claim worth pinning: the sk5 event moved fields
    /// 1-2 and *added* the dequant component and the extent-1 mask fix, none of
    /// which these cells carry, so the rest of the token must not have moved.
    ///
    /// The retired family code (`bin`) is not compared against the class code:
    /// the old code was a caller's label and the new one is derived from the
    /// cell, so equality of the two would be a coincidence, not a claim.
    #[test]
    fn non_gem_tokens_tail_identical_to_the_sk2_era_battery() {
        // (operands, sk2 token, expected sk5 token)
        let cases: [(Vec<FdxOperandDesc>, &str, &str); 4] = [
            (
                vec![f32c(&[4096]), f32c(&[4096]), f32c(&[4096])],
                "sk2|bin|f32|cuda:sm89|ix32|grid|r1|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-",
                "sk5|ew2|f32|cuda:sm89|ix32|grid|r1|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-",
            ),
            (
                vec![f32c(&[7])],
                "sk2|bin|f32|cuda:sm89|ix32|warp|r1|co/00/v1/da/f|-",
                "sk5|ew1|f32|cuda:sm89|ix32|warp|r1|co/00/v1/da/f|-",
            ),
            (
                vec![co(&[4096], DType::I16)],
                "sk2|bin|s16|cuda:sm89|ix32|grid|r1|co/00/v8/d16/f|-",
                "sk5|ew1|i16|cuda:sm89|ix32|grid|r1|co/00/v8/d16/f|-",
            ),
            (
                vec![f32c(&[128, 256])],
                "sk2|bin|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f|-",
                "sk5|ew1|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f|-",
            ),
        ];
        for (ops, sk2, expect) in cases {
            let token = derive(DeclaredFacts::none(), &ops, "cuda:sm89").expect("derives");
            assert_eq!(token, expect);
            let tail = |t: &str| t.split('|').skip(3).collect::<Vec<_>>().join("|");
            assert_eq!(
                tail(&token),
                tail(sk2),
                "fields 4.. moved between the sk2-era battery and sk5"
            );
            let dt = |t: &str| t.split('|').nth(2).unwrap().to_string();
            assert_eq!(dt(&token), dt(sk2).replace("s16", "i16"));
        }
    }

    // ---- (b) the ctr 6/7-component contraction group --------------------

    /// KISS Appendix A.1 dense GEMM skinny-decode cell
    /// `[8,4096]·[4096,4096]→[8,4096]`, f32, non-batched, bit-stable — the sk4
    /// precision group is `/f32/f32/f32/st`. Byte-for-byte the staged golden
    /// (`a1_dense_contraction_cuda` / `a1_dense_contraction_vulkan_target`).
    #[test]
    fn kiss_a1_gem_skinny_decode_golden() {
        let ops = [f32c(&[8, 4096]), f32c(&[4096, 4096]), f32c(&[8, 4096])];
        let cell = DeclaredFacts::contraction(gem_f32(8, 4096, 4096));
        let cuda = derive(cell, &ops, "cuda:sm89").expect("derives");
        assert_eq!(
            cuda,
            "sk5|ctr|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/f32/f32/f32/st"
        );
        // The same cell for a Vulkan target is a different cell (byte-exact
        // target rule, §6.8-0002).
        let vk = derive(cell, &ops, "vulkan:spirv1.6").expect("derives");
        assert_eq!(
            vk,
            "sk5|ctr|f32|vulkan:spirv1.6|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/f32/f32/f32/st"
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
        let cell = DeclaredFacts::contraction(gem_f32(8, 4096, 4096));

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
            let token = derive(cell, &ops, target)
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
            derive(cell, &ops, "vulkansg64.ops-abr.arith-f16.cm-none"),
            None,
            "a non-namespaced target must decline, not pass through"
        );
    }

    /// KISS-lane finding D10 (2026-10-01): the target check used to be bare
    /// `target.contains(':')`, which only asks "does a colon appear
    /// anywhere" — so a target with a second colon, an empty namespace, an
    /// empty vocabulary half, or a forbidden character riding along with a
    /// lone valid-looking colon all passed. §6.8-0001/-0005 want EXACTLY one
    /// `:` separating two non-empty parts with no `|`/`;`/`/`/whitespace/
    /// control characters anywhere in the string.
    #[test]
    fn malformed_namespaced_targets_decline() {
        let ops = [f32c(&[8, 4096]), f32c(&[4096, 4096]), f32c(&[8, 4096])];
        let cell = DeclaredFacts::contraction(gem_f32(8, 4096, 4096));
        for bad in [
            "a:b:c",       // more than one ':'
            ":x",          // empty namespace
            "x:",          // empty vocabulary half
            "cuda:sm|89",  // forbidden '|' rides with an otherwise-valid colon
            "cuda:sm;89",  // forbidden ';'
            "cuda:sm/89",  // forbidden '/'
            "cuda: sm89",  // whitespace
            "cuda:sm89\t", // control character
        ] {
            assert_eq!(
                derive(cell, &ops, bad),
                None,
                "malformed target `{bad:?}` must decline, not pass bare contains(':') check",
            );
        }
    }

    /// KISS-lane finding D11 (2026-10-01): no cap existed on the emitted
    /// token's length before this fix — §6.4-0004 bounds it at 4096 bytes,
    /// and a cell whose target string alone exceeds that must decline rather
    /// than emit an oversized token.
    #[test]
    fn an_oversized_token_declines_rather_than_emitting_unbounded() {
        let ops = [f32c(&[4, 8])];
        let cell = DeclaredFacts::none();
        let huge_target = format!("cuda:{}", "a".repeat(MAX_TOKEN_BYTES));
        assert_eq!(
            derive(cell, &ops, &huge_target),
            None,
            "a token whose target alone exceeds MAX_TOKEN_BYTES must decline",
        );
        // Positive control: the same cell with a short target still derives,
        // so the decline above is about SIZE, not something else about the cell.
        assert!(
            derive(cell, &ops, "cuda:sm89").is_some(),
            "positive control: the cell itself is derivable with a short target",
        );
    }

    /// A batched gem cell carries the conditionally-present `b<class>` right
    /// after `<kdiv>` (7 `/`-parts); the non-batched twin omits it entirely
    /// (6 parts). Byte-for-byte the staged `sk4_gem_batched_cell` golden.
    #[test]
    fn ctr_batched_cell_golden() {
        let ops = [f32c(&[256, 4096]), f32c(&[4096, 4096]), f32c(&[256, 4096])];
        let batched = GemCell {
            batch: Some(256),
            ..gem_f32(256, 4096, 4096)
        };
        let token =
            derive(DeclaredFacts::contraction(batched), &ops, "cuda:sm90").expect("derives");
        assert_eq!(
            token,
            "sk5|ctr|f32|cuda:sm90|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|cmll/d16/bm/f32/f32/f32/st"
        );
        // The non-batched twin differs exactly by the absent /bm coordinate.
        let plain = derive(
            DeclaredFacts::contraction(gem_f32(256, 4096, 4096)),
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
    fn ctr_simt_f32_vs_tf32_distinct_by_mp() {
        let ops = [f32c(&[8, 4096]), f32c(&[4096, 4096]), f32c(&[8, 4096])];
        let simt = derive(
            DeclaredFacts::contraction(gem_f32(8, 4096, 4096)),
            &ops,
            "cuda:sm90",
        )
        .expect("derives");
        let tf32_cell = GemCell {
            math_precision: GemMathPrecision::ReducedMantissa,
            ..gem_f32(8, 4096, 4096)
        };
        let tf32 =
            derive(DeclaredFacts::contraction(tf32_cell), &ops, "cuda:sm90").expect("derives");
        assert_eq!(
            simt,
            "sk5|ctr|f32|cuda:sm90|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/f32/f32/f32/st"
        );
        assert_eq!(
            tf32,
            "sk5|ctr|f32|cuda:sm90|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/f32/f32/f32/rm"
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
    fn ctr_mixed_precision_fp8_golden() {
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
        let token = derive(DeclaredFacts::contraction(cell), &ops, "cuda:sm90").expect("derives");
        assert_eq!(
            token,
            "sk5|ctr|f8e4m3fn|cuda:sm90|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/f8e4m3fn/f32/f16/st"
        );
        // The f32-out twin is a DISTINCT token (the sk2 collision resolved
        // in-key by the precision coordinates, §6.6-0018).
        let twin = GemCell {
            out_dtype: DType::F32,
            ..cell
        };
        let twin_token =
            derive(DeclaredFacts::contraction(twin), &ops, "cuda:sm90").expect("derives");
        assert_ne!(
            token, twin_token,
            "mixed-precision FP8 cells must not collide under sk5"
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
            derive(DeclaredFacts::contraction(mx_weight), &ops, "cuda:sm89"),
            None
        );
        let negative_m = gem_f32(-1, 4096, 4096);
        assert_eq!(
            derive(DeclaredFacts::contraction(negative_m), &ops, "cuda:sm89"),
            None
        );
        let negative_batch = GemCell {
            batch: Some(-2),
            ..gem_f32(8, 4096, 4096)
        };
        assert_eq!(
            derive(
                DeclaredFacts::contraction(negative_batch),
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
            let token = derive(
                DeclaredFacts::contraction(gem_f32(m, n, k)),
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
            if let Some(token) = derive(DeclaredFacts::none(), &[co(&[4096], dt)], "cuda:sm89") {
                assert_clean(&token);
            }
            // Every gem precision position at once.
            let cell = GemCell {
                weight_dtype: dt,
                acc_dtype: dt,
                out_dtype: dt,
                ..gem_f32(8, 4096, 4096)
            };
            if let Some(token) = derive(
                DeclaredFacts::contraction(cell),
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
        let token = derive(
            DeclaredFacts::none(),
            &[co(&[4096], DType::F8E4M3)],
            "cuda:sm89",
        )
        .expect("f8e4m3fn derives");
        assert!(token.starts_with("sk5|ew1|f8e4m3fn|"), "got {token}");
    }

    /// **Corrected 2026-09-26, re-cut at sk5.** This test used to assert that a
    /// plain cell with an `F8E8M0`/`F8E6M2` operand at position 0 must derive a
    /// key. That was wrong: KISS-CLASSIFY §3.2 ratifies the MX scales as valid
    /// only as a sibling operand, never as the dense/primary dtype a generic
    /// (non-role-hinted) cell reads from operand 0.
    ///
    /// **Three DISTINCT senses of "declines" that must not be collapsed:**
    /// - `F8E8M0` — in the closed vocabulary and **spelled**; declining it at
    ///   operand 0 is a POSITIONAL decline, not "this dtype doesn't exist".
    /// - `F8E6M2` — **reserved at sk5** (KISS #517, §6.1-0013). It declines at
    ///   operand 0 because it is reserved, NOT because of its position, and it
    ///   has **no spelling** from [`sk5_dtype_token`] — while `fuel_ir`'s own
    ///   table still spells it (a real, implemented Fuel dtype). That
    ///   disagreement is the named, deriver-local reservation.
    /// - `F6E2M3`/`F6E3M2`/`F4` — outside the closed set entirely.
    ///
    /// Collapsing these into one assertion would hide a future regression where
    /// a scale becomes spellable-and-emitting again at operand 0.
    ///
    /// The legitimate case — a scale read via a caller-supplied role hint, never
    /// a positional operand read — is `gem_weight_role_discriminator`-shaped and
    /// is covered by the byte-match leg, unaffected by this decline.
    #[test]
    fn mx_scale_and_reserved_dtypes_decline_at_operand_zero_for_distinct_reasons() {
        // (1) F8E8M0: spelled, declined positionally.
        assert_eq!(
            derive(
                DeclaredFacts::none(),
                &[co(&[4096], DType::F8E8M0)],
                "cuda:sm89"
            ),
            None,
            "F8E8M0 at operand 0 must typed-decline — a ratified MX scale, valid \
             only as a sibling operand (KISS-CLASSIFY §3.2)",
        );
        assert!(
            sk5_dtype_token(DType::F8E8M0).is_some(),
            "F8E8M0 must remain a spellable token — the decline above is \
             positional (operand-0 usage), not a vocabulary exclusion",
        );

        // (2) F8E6M2: reserved at sk5 — declined, and NOT spelled here.
        assert_eq!(
            derive(
                DeclaredFacts::none(),
                &[co(&[4096], DType::F8E6M2)],
                "cuda:sm89"
            ),
            None,
            "F8E6M2 is reserved at sk5 (KISS #517) and must typed-decline",
        );
        assert_eq!(
            sk5_dtype_token(DType::F8E6M2),
            None,
            "F8E6M2 has no sk5 spelling: reserved, never emitted",
        );
        // The disagreement with fuel-ir is the point: fuel-ir's table still
        // spells it (it is a real Fuel dtype), so the reservation lives in
        // this module's wrapper and nowhere else. If fuel-ir ever starts
        // declining it too, this stops being a deriver-local fact and the
        // GAP row recording the split must be closed.
        assert_eq!(
            fuel_ir::sk4_token(DType::F8E6M2),
            Some("f8e6m2"),
            "fuel-ir's dtype table moved: the sk5 reservation is no longer \
             deriver-local — reconcile with the GAP row"
        );

        // (3) Negative control: the block-scoped sub-byte ELEMENT formats are
        // outside the closed set entirely (a vocabulary exclusion, not a
        // positional one — `sk5_dtype_token` returns None for these too).
        for dt in [DType::F6E2M3, DType::F6E3M2, DType::F4] {
            assert_eq!(
                derive(DeclaredFacts::none(), &[co(&[4096], dt)], "cuda:sm89"),
                None,
                "{dt:?} is NOT in the §6.1 set and must still typed-decline",
            );
            assert!(
                sk5_dtype_token(dt).is_none(),
                "{dt:?} is a vocabulary exclusion, not a positional one",
            );
        }
    }

    /// **`f8e6m2` is reserved at sk5, and the reservation holds in EVERY keyed
    /// dtype position, not only operand 0** — the declines above would pass if
    /// only the primary-dtype path were guarded. A `ctr` cell's three
    /// coordinates and the `(acc + mp)` accumulator are all caller-supplied
    /// `DType`s, each of which funnels through [`sk5_dtype_token`]; this pins
    /// that none of them emits the reserved spelling.
    ///
    /// Positive control per position: the same cell with `f32` in that slot
    /// derives, so the decline is about the dtype, not the cell.
    #[test]
    fn f8e6m2_is_reserved_at_sk5_in_every_keyed_position() {
        let ops = [f32c(&[8, 4096]), f32c(&[4096, 4096]), f32c(&[8, 4096])];
        for (slot, with_reserved) in [
            (
                "weight",
                GemCell {
                    weight_dtype: DType::F8E6M2,
                    ..gem_f32(8, 4096, 4096)
                },
            ),
            (
                "accumulator",
                GemCell {
                    acc_dtype: DType::F8E6M2,
                    ..gem_f32(8, 4096, 4096)
                },
            ),
            (
                "output",
                GemCell {
                    out_dtype: DType::F8E6M2,
                    ..gem_f32(8, 4096, 4096)
                },
            ),
        ] {
            assert!(
                derive(
                    DeclaredFacts::contraction(gem_f32(8, 4096, 4096)),
                    &ops,
                    "cuda:sm90"
                )
                .is_some(),
                "control: the f32 contraction must derive"
            );
            assert_eq!(
                derive(DeclaredFacts::contraction(with_reserved), &ops, "cuda:sm90"),
                None,
                "f8e6m2 in the contraction {slot} coordinate must decline (reserved at sk5)",
            );
        }
        // The non-contraction accumulator coordinate.
        let reduce_ops = [f32c(&[4, 8]), f32c(&[4, 1])];
        assert!(
            derive_acc_mp(
                DeclaredFacts::reduction(ReduceAxes::TrailingAxis),
                &reduce_ops,
                "cuda:sm89",
                Some(AccMp {
                    acc_dtype: DType::F64,
                    math_precision: GemMathPrecision::BitStable,
                }),
            )
            .is_some(),
            "control: an f64 accumulator derives"
        );
        assert_eq!(
            derive_acc_mp(
                DeclaredFacts::reduction(ReduceAxes::TrailingAxis),
                &reduce_ops,
                "cuda:sm89",
                Some(AccMp {
                    acc_dtype: DType::F8E6M2,
                    math_precision: GemMathPrecision::BitStable,
                }),
            ),
            None,
            "f8e6m2 in the (acc + mp) accumulator coordinate must decline",
        );
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
        let plain = derive(DeclaredFacts::reduction(ReduceAxes::All), &ops, "cuda:sm89")
            .expect("plain reduction derives");

        // acc == compute dtype, mp == default ⇒ nothing to declare.
        let diagonal = derive_acc_mp(
            DeclaredFacts::reduction(ReduceAxes::All),
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
            derive_acc_mp(
                DeclaredFacts::reduction(ReduceAxes::All),
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
            derive(DeclaredFacts::contraction(cell), &ops, "cuda:sm89").is_some(),
            "control: the gem cell must derive without an (acc+mp), or the decline below proves nothing about coexistence",
        );
        assert_eq!(
            derive_acc_mp(
                DeclaredFacts::contraction(cell),
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
        // set at sk4 and now emit; see `mx_scale_and_reserved_dtypes_decline_at_operand_zero_for_distinct_reasons`.)
        let bad_dtype = co(&[4096], DType::F4);
        assert_eq!(
            derive(DeclaredFacts::none(), &[bad_dtype], "cuda:sm89"),
            None
        );
        // A non-namespaced target is rejected (§6.8-0001 requires `<ns>:<cap>`).
        assert_eq!(
            derive(DeclaredFacts::none(), &[f32c(&[4096])], "sm89"),
            None
        );
        // No operands → decline.
        assert_eq!(derive(DeclaredFacts::none(), &[], "cuda:sm89"), None);
        // Over MAX_OPERANDS (8) → decline (§6.4-0002).
        let nine = vec![f32c(&[4096]); 9];
        assert_eq!(derive(DeclaredFacts::none(), &nine, "cuda:sm89"), None);
        // A malformed descriptor (shape/strides length mismatch) → decline.
        let mut broken = f32c(&[4096]);
        broken.strides = vec![1, 1];
        assert_eq!(derive(DeclaredFacts::none(), &[broken], "cuda:sm89"), None);
    }

    // ---- reduction cells: §6.5-0009(b) + the KISS A.1 shared vectors --------

    /// A reduced innermost axis derives v1 (§6.5-0009(b)) — the sk2-era
    /// deriver emitted v4 here, diverging from the (unchanged) spec clause.
    #[test]
    fn reduction_vec_width_is_v1_when_innermost_axis_reduced() {
        let token = derive(
            DeclaredFacts::reduction(ReduceAxes::All),
            &[f32c(&[4096])],
            "cuda:sm89",
        )
        .expect("reduction must derive");
        assert_eq!(
            token,
            "sk5|rdc|f32|cuda:sm89|ix32|grid|r1|co/00/v1/d16/f|rall"
        );
    }

    /// A keepdim mask that does NOT cover the innermost axis keeps the
    /// vectorized width (the v1 gate reads the innermost mask bit).
    #[test]
    fn reduction_v1_gate_reads_the_innermost_mask_bit() {
        // rank-2, reducing axis 0 only (mask 0x01): the innermost axis 1 is
        // NOT reduced → v4 stands. The keepdim output `[1,256]` has an own
        // extent-1 axis 0 against frame extent 128, so its broadcast mask is
        // `01` at sk5 (§6.6-0008, #520).
        let token = derive(
            DeclaredFacts::reduction(ReduceAxes::Keepdim(0x01)),
            &[f32c(&[128, 256]), f32c(&[1, 256])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk5|rdc|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/01/v4/d16/f|x01"
        );
    }

    // sk5 note, shared by the reduction goldens below: the keepdim reduction
    // OUTPUT (§6.6-0009 requires keepdim form) has an own axis of extent 1
    // against an iteration-frame extent > 1, so its broadcast mask has that bit
    // set (§6.6-0008, #520): `02` for `[4,1]` in a `[4,8]` frame, `03` for
    // `[1,1]`, `01` for `[1]` in `[8]`, `0a` for `[2,1,3,1]` in `[2,4,3,5]`.
    // KISS `a33282c` published these vectors with `00` there (inconsistent with
    // its own derivation, which Fuel reported); KISS #536 (`5b28bf51`)
    // regenerated them with exactly these masks.

    /// KISS A.1: reduction keepdim `[4,8] → [4,1]` (trailing-axis ⇒ `rlast`).
    #[test]
    fn kiss_a1_reduction_trailing_axis_golden() {
        let token = derive(
            DeclaredFacts::reduction(ReduceAxes::TrailingAxis),
            &[f32c(&[4, 8]), f32c(&[4, 1])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk5|rdc|f32|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/02/v1/da/f|rlast"
        );
    }

    /// KISS A.1: reduction keepdim `[4,8] → [1,1]` (all-axes ⇒ `rall`).
    #[test]
    fn kiss_a1_reduction_all_axes_golden() {
        let token = derive(
            DeclaredFacts::reduction(ReduceAxes::All),
            &[f32c(&[4, 8]), f32c(&[1, 1])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk5|rdc|f32|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/03/v1/da/f|rall"
        );
    }

    /// KISS A.1: rank-1 reduction `[8] → [1]` — the §6.6-0009 tiebreak encodes
    /// `rall`, never `rlast`.
    #[test]
    fn kiss_a1_reduction_rank1_all_axes_golden() {
        let token = derive(
            DeclaredFacts::reduction(ReduceAxes::All),
            &[f32c(&[8]), f32c(&[1])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk5|rdc|f32|cuda:sm89|ix32|warp|r1|co/00/v1/d8/f;co/01/v1/da/f|rall"
        );
    }

    /// KISS A.1: rank-4 reduction over axes 1 and 3 ⇒ explicit keepdim
    /// bitmask `x0a`, work class `block`.
    #[test]
    fn kiss_a1_reduction_keepdim_mask_golden() {
        let token = derive(
            DeclaredFacts::reduction(ReduceAxes::Keepdim(0x0a)),
            &[f32c(&[2, 4, 3, 5]), f32c(&[2, 1, 3, 1])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk5|rdc|f32|cuda:sm89|ix32|block|r4|co/00/v1/da/f;co/0a/v1/da/f|x0a"
        );
    }

    /// KISS-lane finding D7 (2026-10-01): `reduce_field` used to trust the
    /// caller's `ReduceAxes` tag verbatim instead of re-deriving the
    /// classification from `(rank, bitmask)` — so a caller that (legitimately
    /// or not) described a rank-1 reduction via `TrailingAxis` rather than
    /// `All` got `rlast` instead of `rall`, even though
    /// [`kiss_a1_reduction_rank1_all_axes_golden`]'s own doc comment already
    /// states the §6.6-0009 rule ("`rall`, never `rlast`") — that test just
    /// happened to exercise it only through the `All` constructor, which was
    /// never the vulnerable path. `structure_key` canonicalizes by what the
    /// OPERATION is, not by which Rust variant a caller picked to describe
    /// it: two callers describing the identical rank-1 reduction must get the
    /// byte-identical token.
    #[test]
    fn trailing_axis_at_rank1_is_rall_not_rlast() {
        let token = derive(
            DeclaredFacts::reduction(ReduceAxes::TrailingAxis),
            &[f32c(&[8]), f32c(&[1])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token, "sk5|rdc|f32|cuda:sm89|ix32|warp|r1|co/00/v1/d8/f;co/01/v1/da/f|rall",
            "a rank-1 TrailingAxis reduction is the same operation as All at rank 1 \
             and must derive the same reduce field",
        );
    }

    /// D7, second case: a `Keepdim` bitmask that happens to span every axis
    /// (here `0x03` at rank 2 — bits 0 and 1 both set) must canonicalize to
    /// `rall`, not the literal bitmask spelling `x03` — the bitmask form is
    /// reserved for genuinely PARTIAL axis sets (§6.6-0009's `Keepdim` doc:
    /// "for any OTHER axis set").
    #[test]
    fn keepdim_spanning_every_axis_is_rall_not_the_bitmask() {
        let token = derive(
            DeclaredFacts::reduction(ReduceAxes::Keepdim(0x03)),
            &[f32c(&[4, 8]), f32c(&[1, 1])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token, "sk5|rdc|f32|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/03/v1/da/f|rall",
            "Keepdim(0x03) at rank 2 reduces every axis, same operation as All",
        );
    }

    /// D7, third case: an EMPTY `Keepdim` bitmask (no axis reduced) is not a
    /// reduction at all — §6.6-0009/§6.7-0005 the reduce field is `-` when
    /// inapplicable, and a reduction that reduces nothing is exactly that,
    /// never a literal `x00`. **At sk5 it is also not class `rdc`** (§6.5-0015
    /// rule 3 needs a NON-EMPTY set; §6.6-0017 makes the reduce field non-`-`
    /// iff `rdc`), so the cell derives by shape instead.
    #[test]
    fn keepdim_empty_mask_is_dash_not_x00() {
        let token = derive(
            DeclaredFacts::reduction(ReduceAxes::Keepdim(0x00)),
            &[f32c(&[4, 8]), f32c(&[4, 8])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token, "sk5|ew1|f32|cuda:sm89|ix32|warp|r2|co/00/v4/d8/f;co/00/v4/d8/f|-",
            "an empty reduced-axis set is not a reduction (§6.5-0015 rule 3): the cell \
             derives the class its shape gives it (`ew1`, one input pointwise), never \
             `rdc` with a `-` reduce field (the reader's `RdcWithoutReduce` decline), and \
             never x00. Vec width is v4 here, not v1, because nothing is reduced — the v1 \
             gate is specifically for a reduced innermost axis, which an empty mask never is",
        );
    }

    /// At a genuine rank-0 reduction cell (every operand scalar, since frame
    /// rank is the max over operand ranks), `ReduceAxes::All` and
    /// `ReduceAxes::TrailingAxis` must derive byte-identical tokens to the
    /// logically-equivalent `ReduceAxes::Keepdim(0)` -- `reduce_field`
    /// already treats all three as the same empty-mask case.
    ///
    /// Note on what this test does NOT demonstrate: a Sourcery review
    /// described the pre-fix divergence as an OBSERVABLE wire-token
    /// difference ("`All` emits `v1`, `Keepdim(0)` emits `v4`"). Checked
    /// directly against the pre-fix code before trusting that: both
    /// already produced IDENTICAL `v1` tokens at rank 0, because
    /// `operand_sub_key`'s `inner_stride != Some(1)` branch independently
    /// forces `v1` for any rank-0 operand (it has no inner stride at all),
    /// regardless of `innermost_reduced`. So the specific "emits v4" claim
    /// was wrong -- but the underlying concern was real: `innermost_reduced`
    /// WAS computing a logically wrong value at rank 0 (`true` for
    /// `All`/`TrailingAxis` when the reduced set is actually empty), it
    /// just happened not to reach the wire through this particular
    /// downstream guard. A future change to `operand_sub_key`'s OR-chain
    /// (e.g. relaxing the inner-stride check) could have made that latent
    /// wrongness live. This test guards the now-provably-correct derivation
    /// rather than a wire-level regression that never existed.
    #[test]
    fn rank0_all_and_keepdim_zero_derive_identical_tokens() {
        let all_token = derive(
            DeclaredFacts::reduction(ReduceAxes::All),
            &[f32c(&[]), f32c(&[])],
            "cuda:sm89",
        )
        .expect("derives");
        let trailing_token = derive(
            DeclaredFacts::reduction(ReduceAxes::TrailingAxis),
            &[f32c(&[]), f32c(&[])],
            "cuda:sm89",
        )
        .expect("derives");
        let keepdim_token = derive(
            DeclaredFacts::reduction(ReduceAxes::Keepdim(0x00)),
            &[f32c(&[]), f32c(&[])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(all_token, keepdim_token, "All vs Keepdim(0) at rank 0");
        assert_eq!(
            trailing_token, keepdim_token,
            "TrailingAxis vs Keepdim(0) at rank 0"
        );
    }

    /// D7, exhaustive: for every `(rank, bitmask)` pair the three canonical
    /// forms are mutually exclusive and exhaustive over `Keepdim`'s own
    /// output space — this is the regression guard against the class of bug
    /// above recurring in a shape the three named cases don't happen to hit.
    #[test]
    fn reduce_field_keepdim_classification_is_exhaustive_over_rank_and_mask() {
        for rank in 0usize..=8 {
            let full_mask: u8 = if rank == 0 {
                0
            } else if rank >= 8 {
                0xFF
            } else {
                (1u8 << rank) - 1
            };
            for m in 0u32..=0xFF {
                let m = m as u8;
                // Keepdim's own contract: a bit set above `rank` is not a
                // representable axis — only exercise in-range masks.
                if m & !full_mask != 0 {
                    continue;
                }
                let field = reduce_field(ReduceAxes::Keepdim(m).mask(rank), rank);
                if m == 0 {
                    assert_eq!(field, "-", "rank {rank} mask {m:#04x} must be '-'");
                } else if rank > 0 && m == full_mask {
                    assert_eq!(field, "rall", "rank {rank} mask {m:#04x} must be 'rall'");
                } else if rank > 0 && m == (1u8 << (rank - 1)) {
                    assert_eq!(field, "rlast", "rank {rank} mask {m:#04x} must be 'rlast'");
                } else {
                    assert_eq!(
                        field,
                        format!("x{m:02x}"),
                        "rank {rank} mask {m:#04x} must be the literal bitmask",
                    );
                }
            }
        }
    }

    // ---- §6.5/§6.6 derivation pins (spec-conformance fixes) -----------------

    /// The innermost axis is axis rank−1 (§6.3-0011) even when its extent is
    /// 1: a `[4,1]` operand buckets `da` and derives v1 (the sk2-era
    /// rposition(extent>1) inner axis read extent 4 ⇒ d4/v4).
    #[test]
    fn trailing_unit_axis_reads_rank_minus_1_inner() {
        let token = derive(DeclaredFacts::none(), &[f32c(&[4, 1])], "cuda:sm89").expect("derives");
        assert_eq!(token, "sk5|ew1|f32|cuda:sm89|ix32|warp|r2|co/00/v1/da/f|-");
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
        let token = derive(DeclaredFacts::none(), &[f32c(&[0])], "cuda:sm89")
            .expect("derives (never panics)");
        assert_eq!(token, "sk5|ew1|f32|cuda:sm89|ix32|warp|r1|co/00/v1/da/f|-");
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
        let token = derive(DeclaredFacts::none(), &[desc], "cuda:sm89").expect("derives");
        assert_eq!(token, "sk5|ew1|f32|cuda:sm89|ix32|warp|r2|co/00/v1/da/r|-");
    }

    /// `alignment = 0` (unspecified base) cannot honor a packed load ⇒ v1
    /// (§6.5-0009; the sk2-era `0 % vbytes == 0` would have vectorized).
    #[test]
    fn alignment_zero_derives_v1() {
        let mut desc = f32c(&[4096]);
        desc.align_bytes = 0;
        let token = derive(DeclaredFacts::none(), &[desc], "cuda:sm89").expect("derives");
        assert_eq!(token, "sk5|ew1|f32|cuda:sm89|ix32|grid|r1|co/00/v1/d16/f|-");
    }

    /// Work class and per-operand masks read the ITERATION FRAME
    /// (§6.5-0010 / §6.6-0013): a rank-deficient operand-0 is right-aligned,
    /// its missing frame axis counted as broadcast for the admissibility
    /// MASK (stride 0) — so the cell is `grid` (frame 128·256), not `block`
    /// (operand-0's own 256), and operand-0's mask is `01`. Its `layout_tag`
    /// is `co`, not `br`, per §6.5-0014 scope C (KISS#519): layout reads the
    /// operand's OWN dense `[256]` shape/strides, not the padded view —
    /// padding is an admissibility fact (the mask), not a statement about
    /// this operand's own memory layout. Same reasoning gives it `v4`, not
    /// `v1`: `mask != 0` alone no longer forces scalar width.
    #[test]
    fn work_class_and_masks_use_the_iteration_frame() {
        let token = derive(
            DeclaredFacts::none(),
            &[f32c(&[256]), f32c(&[128, 256]), f32c(&[128, 256])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk5|ew2|f32|cuda:sm89|ix32|grid|r2|co/01/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-"
        );
    }

    /// §6.5-0014 scope C (KISS#519) worked example: an EXPLICIT `[1,256]`
    /// operand with a stride-0 leading axis of extent 1 derives the SAME
    /// `co/01/v4/d16/f` as the rank-deficient `[256]` operand above --
    /// `layout_tag` only reports `br` for an OWN axis with extent > 1 AND
    /// stride 0 (§6.5-0002 step 1); extent-1 axes are excluded from the
    /// contiguity product entirely, so a stride-0 extent-1 axis is `co`,
    /// not `br`, matching the rank-deficient case's own-axes view exactly.
    #[test]
    fn explicit_unit_broadcast_axis_matches_rank_deficient_own_axes_layout() {
        let explicit = FdxOperandDesc::from_layout(
            &Layout::new(
                Shape::from(vec![1usize, 256]),
                [0isize, 1].into_iter().collect::<StrideVec>(),
                0,
            ),
            DType::F32,
        );
        let token = derive(
            DeclaredFacts::none(),
            &[explicit, f32c(&[128, 256]), f32c(&[128, 256])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk5|ew2|f32|cuda:sm89|ix32|grid|r2|co/01/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-"
        );
    }

    /// §6.5-0014 scope C (KISS#519) worked example: a dense vs. a strided
    /// rank-3 operand in a rank-4 frame, differing ONLY in their own
    /// strides -- layout_tag (and therefore `vec`) must track that
    /// difference, which a frame-padded view (identical `ext_p` for both)
    /// could never distinguish.
    #[test]
    fn frame_padded_dense_vs_strided_rank3_differ_by_own_layout() {
        let dense = FdxOperandDesc::from_layout(
            &Layout::new(
                Shape::from(vec![4usize, 16, 64]),
                [1024isize, 64, 1].into_iter().collect::<StrideVec>(),
                0,
            ),
            DType::F32,
        );
        let strided = FdxOperandDesc::from_layout(
            &Layout::new(
                Shape::from(vec![4usize, 16, 64]),
                [2048isize, 128, 2].into_iter().collect::<StrideVec>(),
                0,
            ),
            DType::F32,
        );
        let frame_setter = f32c(&[8, 4, 16, 64]);

        let dense_token = derive(
            DeclaredFacts::none(),
            &[dense, frame_setter.clone()],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            dense_token,
            "sk5|ew1|f32|cuda:sm89|ix32|grid|r4|co/01/v4/d16/f;co/00/v4/d16/f|-"
        );

        let strided_token =
            derive(DeclaredFacts::none(), &[strided, frame_setter], "cuda:sm89").expect("derives");
        assert_eq!(
            strided_token, "sk5|ew1|f32|cuda:sm89|ix32|grid|r4|st/01/v1/d16/f;co/00/v4/d16/f|-",
            "a non-forward-unit own stride must derive `st`/`v1` even though \
             the frame-padded extents are byte-identical to the dense case"
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
        let token = derive(
            DeclaredFacts::none(),
            &[f32c(&[128, 256]), bcast, f32c(&[128, 256])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk5|ew2|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;br/01/v1/d16/f;co/00/v4/d16/f|-"
        );
    }

    /// KISS A.1: the canonical rank-2 binary cell (two inputs + one output) and
    /// the in-place accumulate cell (2 operands — the read-modify-write operand
    /// appears exactly once, §6.6-0014, classified as an INPUT) — byte-for-byte
    /// the staged `a1_elementwise_binary_canonical` / `a1_binary_two_operands`
    /// goldens. Both derive `ew2`: two input operands, pointwise by shape.
    #[test]
    fn kiss_a1_bin_canonical_and_inplace_goldens() {
        let op = f32c(&[128, 256]);
        let canonical = derive(
            DeclaredFacts::none(),
            &[op.clone(), op.clone(), op.clone()],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            canonical,
            "sk5|ew2|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-"
        );
        // Explicit roles: `acc += x` — input, then the in-place accumulator.
        let inplace = derive_structure_key_token(
            &DeclaredFacts::none(),
            &[
                CellOperand::input(op.clone()),
                CellOperand::in_place(op.clone()),
            ],
            "cuda:sm89",
            None,
        )
        .expect("derives");
        assert_eq!(
            inplace,
            "sk5|ew2|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f|-"
        );
        // The same two descriptors as input + OUTPUT are a different cell: one
        // input, so `ew1`. The in-place role is what makes it `ew2`.
        let in_out =
            derive(DeclaredFacts::none(), &[op.clone(), op], "cuda:sm89").expect("derives");
        assert_eq!(
            in_out,
            "sk5|ew1|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f|-"
        );
    }

    /// KISS A.1: unary elementwise f16 `[64,128]` derives v8 (2-byte dtype) —
    /// byte-for-byte the staged `a1_unary_f16_v8` golden modulo op family
    /// (Fuel's category enum has no `une`; the operand derivation is shared,
    /// so the f16/v8 sub-keys are pinned via a `bin` cell).
    #[test]
    fn f16_wide_vector_matches_kiss_a1_subkeys() {
        let op = co(&[64, 128], DType::F16);
        let token = derive(DeclaredFacts::none(), &[op.clone(), op], "cuda:sm89").expect("derives");
        assert_eq!(
            token,
            "sk5|ew1|f16|cuda:sm89|ix32|grid|r2|co/00/v8/d16/f;co/00/v8/d16/f|-"
        );
    }

    // ---- sk5: the derived cell class (§6.5-0006, §6.5-0015) -----------------

    fn class_of(facts: DeclaredFacts, descs: &[FdxOperandDesc]) -> Option<CellClass> {
        derive_cell_class(&facts, &canon(descs))
    }

    /// §6.5-0015: **the first rule that holds decides**, and the three declared
    /// facts are tried in the order ctr → ndx → rdc *before* any shape test.
    /// Every adjacent pair is exercised with BOTH facts declared, so a swapped
    /// order fails on the pair it swapped; the single-fact cells are the
    /// controls that show each fact does produce its own class.
    #[test]
    fn declared_facts_are_tried_in_rule_order_before_the_shape_test() {
        let same = [f32c(&[8, 16]), f32c(&[8, 16]), f32c(&[8, 16])];
        let gem = gem_f32(8, 16, 16);

        // Controls: each fact alone derives its own class.
        assert_eq!(
            class_of(DeclaredFacts::contraction(gem), &same),
            Some(CellClass::DenseContraction)
        );
        assert_eq!(
            class_of(DeclaredFacts::indexed(1), &same),
            Some(CellClass::IndexedAccess)
        );
        assert_eq!(
            class_of(DeclaredFacts::reduction(ReduceAxes::All), &same),
            Some(CellClass::Reduction)
        );
        assert_eq!(
            class_of(DeclaredFacts::none(), &same),
            Some(CellClass::ElementwiseTwo)
        );

        // ctr beats ndx and rdc.
        let mut f = DeclaredFacts::contraction(gem);
        f.index_operand = Some(1);
        f.reduced_axes = Some(ReduceAxes::All);
        assert_eq!(class_of(f, &same), Some(CellClass::DenseContraction));

        // ndx beats rdc.
        let mut f = DeclaredFacts::indexed(1);
        f.reduced_axes = Some(ReduceAxes::All);
        assert_eq!(class_of(f, &same), Some(CellClass::IndexedAccess));
    }

    /// Rule 4/5: pointwise by shape, classified by INPUT-operand count. 1/2/3
    /// inputs are `ew1`/`ew2`/`ew3`; 0 inputs and 4+ inputs are `oth`
    /// (§6.5-0015 rule 5 names both).
    #[test]
    fn pointwise_cells_classify_by_input_count_and_only_one_to_three() {
        let d = || f32c(&[8, 16]);
        let ops = |n_in: usize, n_out: usize| -> Vec<CellOperand> {
            (0..n_in)
                .map(|_| CellOperand::input(d()))
                .chain((0..n_out).map(|_| CellOperand::output(d())))
                .collect()
        };
        let cls = |n_in, n_out| derive_cell_class(&DeclaredFacts::none(), &ops(n_in, n_out));
        assert_eq!(cls(1, 1), Some(CellClass::ElementwiseOne));
        assert_eq!(cls(2, 1), Some(CellClass::ElementwiseTwo));
        assert_eq!(cls(3, 1), Some(CellClass::ElementwiseThree));
        assert_eq!(cls(4, 1), Some(CellClass::Other), "four inputs derive oth");
        assert_eq!(
            cls(0, 2),
            Some(CellClass::Other),
            "no input operand derives oth"
        );
        // A second OUTPUT does not change the input count (§6.6-0014).
        assert_eq!(cls(2, 2), Some(CellClass::ElementwiseTwo));
    }

    /// Rule 4's pointwise predicate has two halves and BOTH are load-bearing:
    /// (a) every operand's extent at every frame axis is 1 or the frame extent;
    /// (b) every WRITTEN operand has exactly the frame extent at every axis AND
    /// the iteration rank. Each half gets a cell that violates only it.
    #[test]
    fn a_cell_that_is_not_pointwise_by_shape_derives_oth() {
        // (a) an input extent that is neither 1 nor the frame extent: frame is
        // [4,8] (from the output); the input's axis-1 extent 3 is not.
        let bad_input = [f32c(&[4, 3]), f32c(&[4, 8])];
        assert_eq!(
            class_of(DeclaredFacts::none(), &bad_input),
            Some(CellClass::Other)
        );

        // (b1) the written operand lacks the frame extent: input [128,256],
        // output [64,256] — the corpus's `cell_class_oth_not_pointwise` shape.
        let short_output = [f32c(&[128, 256]), f32c(&[64, 256])];
        assert_eq!(
            class_of(DeclaredFacts::none(), &short_output),
            Some(CellClass::Other)
        );

        // (b2) the written operand has the frame extents but NOT the iteration
        // rank: input [4,8] against a rank-1 output [8]. Every extent check in
        // (a) passes (the missing axis reads as 1), so only the rank clause
        // sees it.
        let low_rank_output = [f32c(&[4, 8]), f32c(&[8])];
        assert_eq!(
            class_of(DeclaredFacts::none(), &low_rank_output),
            Some(CellClass::Other),
            "a written operand must have the iteration RANK, not just extents"
        );

        // Control for all three: the same input against a full-frame output is
        // pointwise.
        let ok = [f32c(&[4, 8]), f32c(&[4, 8])];
        assert_eq!(
            class_of(DeclaredFacts::none(), &ok),
            Some(CellClass::ElementwiseOne)
        );
    }

    /// An INPUT may be broadcast (extent 1, or an axis it does not have) and
    /// the cell is still pointwise; a WRITTEN operand may not be.
    #[test]
    fn broadcast_inputs_are_pointwise_but_a_broadcast_written_operand_is_not() {
        // Input [256] and [1,256] against a [128,256] output: still `ew2`.
        let bcast = [f32c(&[256]), f32c(&[1, 256]), f32c(&[128, 256])];
        assert_eq!(
            class_of(DeclaredFacts::none(), &bcast),
            Some(CellClass::ElementwiseTwo)
        );
        // An in-place operand counts as written: a [1,256] accumulator in a
        // [128,256] frame has extent 1 where the frame has 128, so the cell is
        // NOT pointwise.
        let ops = [
            CellOperand::input(f32c(&[128, 256])),
            CellOperand::in_place(f32c(&[1, 256])),
        ];
        assert_eq!(
            derive_cell_class(&DeclaredFacts::none(), &ops),
            Some(CellClass::Other)
        );
        // Control: the same accumulator at the frame extent is `ew2`, because
        // an in-place operand counts ONCE, as an input (§6.6-0014).
        let ok = [
            CellOperand::input(f32c(&[128, 256])),
            CellOperand::in_place(f32c(&[128, 256])),
        ];
        assert_eq!(
            derive_cell_class(&DeclaredFacts::none(), &ok),
            Some(CellClass::ElementwiseTwo)
        );
    }

    /// §6.6-0014: an input after an output is not canonical order. The deriver
    /// declines — `None`, never a class — and the token path declines with it.
    #[test]
    fn a_cell_not_in_canonical_order_declines() {
        let d = f32c(&[8, 16]);
        let out_first = [
            CellOperand::output(d.clone()),
            CellOperand::input(d.clone()),
        ];
        assert_eq!(derive_cell_class(&DeclaredFacts::none(), &out_first), None);
        assert_eq!(
            derive_structure_key_token(&DeclaredFacts::none(), &out_first, "cuda:sm89", None),
            None
        );
        // Control: the same two operands in canonical order derive.
        let in_first = [CellOperand::input(d.clone()), CellOperand::output(d)];
        assert!(
            derive_structure_key_token(&DeclaredFacts::none(), &in_first, "cuda:sm89", None)
                .is_some()
        );
    }

    /// §6.6-0012: the index role names an operand that is READ. A slot that is
    /// out of range or that names an output is an error, not a role hint.
    #[test]
    fn an_index_slot_must_name_a_read_operand() {
        let d = f32c(&[8, 16]);
        let ops = canon(&[d.clone(), d.clone(), d]); // [in, in, out]
        assert_eq!(
            derive_cell_class(&DeclaredFacts::indexed(1), &ops),
            Some(CellClass::IndexedAccess)
        );
        assert_eq!(
            derive_cell_class(&DeclaredFacts::indexed(2), &ops),
            None,
            "slot 2 is the output"
        );
        assert_eq!(
            derive_cell_class(&DeclaredFacts::indexed(3), &ops),
            None,
            "slot 3 is out of range"
        );
    }

    /// §6.6-0010 / §6.6-0017: the contraction field is present iff `ctr`, and
    /// the reduce field is non-`-` iff `rdc`. A cell that declares BOTH the
    /// contraction roles and a reduced set is `ctr` — it carries the
    /// contraction field and a `-` reduce field, never both gated fields.
    /// Likewise an `ndx` cell that also declares a reduced set.
    #[test]
    fn the_gated_fields_follow_the_derived_class_not_the_declared_facts() {
        let ops = [f32c(&[8, 16]), f32c(&[16, 16]), f32c(&[8, 16])];
        let mut f = DeclaredFacts::contraction(gem_f32(8, 16, 16));
        f.reduced_axes = Some(ReduceAxes::All);
        let ctr = derive(f, &ops, "cuda:sm89").expect("derives");
        let fields: Vec<&str> = ctr.split('|').collect();
        assert_eq!(fields[1], "ctr");
        assert_eq!(fields[8], "-", "reduce field is `-` on a ctr cell: {ctr}");
        assert!(
            fields[9].starts_with("c"),
            "contraction field present: {ctr}"
        );
        assert_eq!(fields.len(), 10);

        let mut g = DeclaredFacts::indexed(0);
        g.reduced_axes = Some(ReduceAxes::TrailingAxis);
        let ndx = derive(g, &ops, "cuda:sm89").expect("derives");
        let fields: Vec<&str> = ndx.split('|').collect();
        assert_eq!(fields[1], "ndx");
        assert_eq!(fields[8], "-", "reduce field is `-` on an ndx cell: {ndx}");
        assert_eq!(fields.len(), 9, "no contraction field on a non-ctr cell");

        // The v1 reduction gate is cell-level and gated on `rdc` too: a ctr
        // cell that merely DECLARES a trailing reduced set must not have its
        // operands forced to scalar width by it.
        assert!(
            ctr.contains("/v4/"),
            "a declared-but-unused reduced set leaked into the v1 gate: {ctr}"
        );
    }

    /// **An empty reduced set is not a reduction** (§6.5-0015 rule 3), at every
    /// rank and by every constructor — and the reader-side decline it would
    /// otherwise become (`RdcWithoutReduce`, `rdc` with a `-` reduce field) is
    /// unreachable on the emit side. Control: a non-empty set at the same shape
    /// IS `rdc`.
    #[test]
    fn an_empty_reduced_set_never_derives_rdc_with_a_dash_reduce_field() {
        for (name, axes, descs) in [
            (
                "keepdim(0) rank 2",
                ReduceAxes::Keepdim(0),
                vec![f32c(&[4, 8]), f32c(&[4, 8])],
            ),
            ("all at rank 0", ReduceAxes::All, vec![f32c(&[]), f32c(&[])]),
            (
                "trailing at rank 0",
                ReduceAxes::TrailingAxis,
                vec![f32c(&[]), f32c(&[])],
            ),
        ] {
            let token = derive(DeclaredFacts::reduction(axes), &descs, "cuda:sm89")
                .unwrap_or_else(|| panic!("{name} must derive"));
            let class = token.split('|').nth(1).unwrap();
            assert_ne!(
                class, "rdc",
                "{name}: derived rdc with an empty set: {token}"
            );
            assert_eq!(token.rsplit('|').next(), Some("-"), "{name}: {token}");
        }
        let control = derive(
            DeclaredFacts::reduction(ReduceAxes::All),
            &[f32c(&[4, 8]), f32c(&[4, 8])],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(control.split('|').nth(1), Some("rdc"), "{control}");
    }

    // ---- sk5: dequant_form, the optional sixth sub-key component ------------

    /// §6.3-0009a / §6.6-0007: an operand that carries a `quant` record gets
    /// `/dl` or `/dc` as the LAST sub-key component; an operand without one is
    /// byte-identical to the five-component form.
    #[test]
    fn dequant_form_is_the_optional_sixth_subkey_component() {
        let descs = [f32c(&[8, 4096]), f32c(&[4096, 4096]), f32c(&[8, 4096])];
        let facts = DeclaredFacts::contraction(gem_f32(8, 4096, 4096));
        let build = |form: Option<DequantForm>| -> String {
            let mut ops = canon(&descs);
            if let Some(f) = form {
                ops[0] = ops[0].clone().with_dequant_form(f);
            }
            derive_structure_key_token(&facts, &ops, "cuda:sm90", None).expect("derives")
        };
        let plain = build(None);
        let linear = build(Some(DequantForm::Linear));
        let codebook = build(Some(DequantForm::Codebook));

        // The component sits on operand 0 only, after `flip`.
        assert!(
            linear.contains("|co/00/v4/d16/f/dl;co/00/v4/d16/f;co/00/v4/d16/f|"),
            "{linear}"
        );
        assert!(
            codebook.contains("|co/00/v4/d16/f/dc;co/00/v4/d16/f;co/00/v4/d16/f|"),
            "{codebook}"
        );
        // An operand with no quant record is byte-identical to the
        // five-component form: stripping the component recovers the plain token.
        assert_eq!(linear.replacen("/dl;", ";", 1), plain);
        assert_eq!(codebook.replacen("/dc;", ";", 1), plain);
        // The rationale made executable: the two forms MUST NOT share a key.
        assert_ne!(linear, codebook);
        assert_ne!(linear, plain);
        assert_ne!(codebook, plain);
    }

    /// The sub-key never has more than the six permitted parts (§6.7-0004's
    /// `BadOperandSubKey` is unrepresentable on emit): a fully quantized cell,
    /// every operand carrying a form, still serializes six parts per operand.
    #[test]
    fn a_sub_key_never_exceeds_six_parts() {
        let descs = [f32c(&[8, 16]), f32c(&[16, 16]), f32c(&[8, 16])];
        let ops: Vec<CellOperand> = canon(&descs)
            .into_iter()
            .map(|o| o.with_dequant_form(DequantForm::Codebook))
            .collect();
        let token = derive_structure_key_token(
            &DeclaredFacts::contraction(gem_f32(8, 16, 16)),
            &ops,
            "cuda:sm89",
            None,
        )
        .expect("derives");
        let sub_keys = token.split('|').nth(7).unwrap();
        for sk in sub_keys.split(';') {
            let parts = sk.split('/').count();
            assert_eq!(parts, 6, "sub-key `{sk}` has {parts} parts: {token}");
            assert!(sk.ends_with("/dc"), "{sk}");
        }
    }

    // ---- sk5: the extent-1 broadcast mask (§6.6-0008, #520) ------------------

    /// **#520:** an own axis of extent 1 counts as stride 0 for the broadcast
    /// mask *whatever stride the descriptor records for it*, so the same tensor
    /// does not spell two tokens. At sk4 `[1,256]` with a non-zero leading
    /// stride left bit 0 clear (`co/00/…`); at sk5 it is set (`co/01/…`) and
    /// equals the stride-0 spelling.
    #[test]
    fn an_extent_one_axis_sets_its_mask_bit_whatever_stride_is_recorded() {
        let unit = |lead_stride: isize| {
            FdxOperandDesc::from_layout(
                &Layout::new(
                    Shape::from(vec![1usize, 256]),
                    [lead_stride, 1].into_iter().collect::<StrideVec>(),
                    0,
                ),
                DType::F32,
            )
        };
        let frame_setter = f32c(&[128, 256]);
        let key = |d: FdxOperandDesc| {
            derive(
                DeclaredFacts::none(),
                &[d, frame_setter.clone(), frame_setter.clone()],
                "cuda:sm89",
            )
            .expect("derives")
        };
        let zero = key(unit(0));
        let nonzero = key(unit(256));
        let odd = key(unit(7));
        assert_eq!(
            nonzero, zero,
            "one tensor, one spelling: the recorded stride of a unit axis is arbitrary"
        );
        assert_eq!(odd, zero);
        assert!(
            zero.contains("|co/01/v4/d16/f;"),
            "bit 0 must be set for the unit axis: {zero}"
        );
    }

    /// The other half of §6.6-0008: the bit is set only where the **frame**
    /// extent is `> 1`. An extent-1 axis in a frame whose extent there is also
    /// 1 sets nothing — otherwise the #520 fix would over-fire on every scalar
    /// -ish operand. This is the sabotage sibling of the test above.
    #[test]
    fn an_extent_one_axis_in_a_unit_frame_axis_sets_no_bit() {
        let unit = FdxOperandDesc::from_layout(
            &Layout::new(
                Shape::from(vec![1usize, 256]),
                [256isize, 1].into_iter().collect::<StrideVec>(),
                0,
            ),
            DType::F32,
        );
        // Frame is [1,256]: axis 0 has frame extent 1.
        let token =
            derive(DeclaredFacts::none(), &[unit.clone(), unit], "cuda:sm89").expect("derives");
        assert!(
            token.contains("|co/00/v4/d16/f;co/00/v4/d16/f|"),
            "frame extent 1 must leave the bit clear: {token}"
        );
    }
}
