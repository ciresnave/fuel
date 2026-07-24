//! Fuel's INDEPENDENT `structure_key` derivation — the second implementation
//! for the two-implementation freeze-gate (KISS-CLASSIFY §6.6/§6.7).
//!
//! This is deliberately **Baracuda-free**: it recomputes the same `sk3` token
//! from Fuel's own [`FdxOperandDesc`] projection, with **no** `baracuda_kernels_*`
//! import, so a byte-match against Baracuda's emitted token is a genuine
//! two-implementation agreement. (K1 opacity — "Fuel never derives the key" —
//! governs the DISPATCH seam in [`super::structure_key`]; the freeze-gate is the
//! deliberate exception: Fuel derives the key independently *to check* it, never
//! to route.)
//!
//! Schema version: **sk3** (KISS-CLASSIFY §6.4-0003 at PR #81). The sk2→sk3
//! delta this deriver implements, derived from the staged spec clauses (not
//! from Baracuda's implementation):
//! - every token re-prefixes `sk2|` → `sk3|` (§6.7-0002, canonical spelling);
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

/// The math-precision key coordinate of a `gem` cell — `<mp>` in the sk3
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
/// extent, and the sk3 precision coordinates — weight / accumulator / output
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

/// The op-family a `structure_key` keys on — the KISS-CLASSIFY §6.5-0006
/// 3-letter domain (the subset Fuel can present today). `Reduction` carries its
/// reduce field (§6.6-0009); `Contraction` carries the sk3 [`GemCell`] role
/// hints + precision coordinates (§6.6-0016 / §6.7-0006) — the former
/// pending-D1 decline is settled by the sk3 schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuelOpCategory {
    BinaryElementwise,
    TernaryElementwise,
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
            FuelOpCategory::BinaryElementwise => "bin",
            FuelOpCategory::TernaryElementwise => "ter",
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

/// Derive the KISS `sk3` `structure_key` token for a cell, independently of
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
    let index_width = if max_touched >= (1i128 << 31) { "ix64" } else { "ix32" };

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

    // Field 9 (gem only) — the sk3 contraction field (§6.7-0006).
    let contraction = match op {
        FuelOpCategory::Contraction(cell) => Some(contraction_field(&cell)?),
        _ => None,
    };

    let mut token = format!(
        "sk3|{op}|{dtype}|{target}|{idx}|{work}|r{rank}|{ops}|{reduce}",
        op = op.code(),
        idx = index_width,
        work = work_class,
        ops = operand_keys.join(";"),
        reduce = op.reduce_field(),
    );
    if let Some(c) = contraction {
        token.push('|');
        token.push_str(&c);
    }
    Some(token)
}

/// The sk3 `gem` contraction field (§6.7-0006):
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
/// closed 22-token sk3 set. The FP8 spelling is variant-explicit: Fuel's
/// `F8E4M3` is the OCP format → `e4m3fn` (the bare `e4m3` spelling is retired
/// at sk3); the reserved `fnuz` variants have no Fuel `DType` and therefore
/// can never be emitted. The MX formats (`F6E2M3`/`F6E3M2`/`F4`/`F8E8M0`) are
/// **not** in the KISS set — Fuel's RFC #9 asks to add them — so they are a
/// typed decline (`None`), never a guessed token. Exhaustive so a new Fuel
/// `DType` is a compile error here, not a silent miss.
fn dtype_token(dt: DType) -> Option<&'static str> {
    Some(match dt {
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::I8 => "s8",
        DType::I16 => "s16",
        DType::U8 => "u8",
        DType::U32 => "u32",
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::F8E4M3 => "e4m3fn",
        // MX element formats — not in the KISS §6.1 closed set (RFC #9 pending).
        DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => return None,
    })
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
            if vbytes <= 16 && o.align_bytes % vbytes == 0 && inner_extent >= l && inner_extent % l == 0 {
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
/// active non-unit axes, innermost first: **(1)** `br` if any axis of extent
/// > 1 has stride 0; **(2)** `co` if each active non-unit axis's `|stride|`
/// equals the running product of the inner active non-unit extents (a fully
/// reversed view is therefore `co` — the reversal lives in the flipped flag);
/// **(3)** `ic` if the innermost active non-unit axis has `|stride| == 1`;
/// **(4)** else `st`. No active axis of extent > 1 ⇒ `co` (empty product).
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
    if let Some(i) = (0..ext.len()).rev().find(|&i| ext[i] > 1) {
        if strides[i].unsigned_abs() == 1 {
            return "ic";
        }
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

    // ---- (a) sk3 prefix on every token class --------------------------------

    /// The relu_add f32 grid-stride freeze-gate cell (condition-1): 3 rank-1
    /// f32 operands [4096], contiguous, offset 0 (align 256): in0, in1, out.
    /// Byte-for-byte the KISS PR #81 staged golden
    /// (`relu_add_generated_r1_cell`).
    #[test]
    fn fuel_derives_relu_add_sk3_token_byte_for_byte() {
        let op = f32c(&[4096]);
        let token = derive_structure_key_token(
            FuelOpCategory::BinaryElementwise,
            &[op.clone(), op.clone(), op],
            "cuda:sm89",
        )
        .expect("relu_add f32 must derive a token");
        assert_eq!(
            token,
            "sk3|bin|f32|cuda:sm89|ix32|grid|r1|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-"
        );
    }

    /// Every derivable op-family token carries the sk3 prefix and no sk2 bytes.
    #[test]
    fn sk3_prefix_on_every_token_class() {
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
            let token = derive_structure_key_token(cat, &[op.clone()], "cuda:sm89")
                .unwrap_or_else(|| panic!("{:?} must derive", cat));
            assert!(token.starts_with("sk3|"), "{token} lacks the sk3 prefix");
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
                "sk3|bin|f32|cuda:sm89|ix32|grid|r1|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-",
            ),
            (
                FuelOpCategory::BinaryElementwise,
                vec![f32c(&[7])],
                // sk2: "sk2|bin|f32|cuda:sm89|ix32|warp|r1|co/00/v1/da/f|-"
                "sk3|bin|f32|cuda:sm89|ix32|warp|r1|co/00/v1/da/f|-",
            ),
            (
                FuelOpCategory::BinaryElementwise,
                vec![co(&[4096], DType::I16)],
                // sk2: "sk2|bin|s16|cuda:sm89|ix32|grid|r1|co/00/v8/d16/f|-"
                "sk3|bin|s16|cuda:sm89|ix32|grid|r1|co/00/v8/d16/f|-",
            ),
            (
                FuelOpCategory::BinaryElementwise,
                vec![f32c(&[128, 256])],
                // sk2: "sk2|bin|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f|-"
                "sk3|bin|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f|-",
            ),
        ];
        for (cat, ops, expect) in cases {
            let token = derive_structure_key_token(cat, &ops, "cuda:sm89").expect("derives");
            assert_eq!(token, expect);
        }
    }

    // ---- (b) the sk3 gem 6/7-component contraction group --------------------

    /// KISS Appendix A.1 dense GEMM skinny-decode cell
    /// `[8,4096]·[4096,4096]→[8,4096]`, f32, non-batched, bit-stable — the sk3
    /// precision group is `/f32/f32/f32/st`. Byte-for-byte the staged golden
    /// (`a1_dense_contraction_cuda` / `a1_dense_contraction_vulkan_target`).
    #[test]
    fn kiss_a1_gem_skinny_decode_golden() {
        let ops = [f32c(&[8, 4096]), f32c(&[4096, 4096]), f32c(&[8, 4096])];
        let cell = FuelOpCategory::Contraction(gem_f32(8, 4096, 4096));
        let cuda = derive_structure_key_token(cell, &ops, "cuda:sm89").expect("derives");
        assert_eq!(
            cuda,
            "sk3|gem|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/f32/f32/f32/st"
        );
        // The same cell for a Vulkan target is a different cell (byte-exact
        // target rule, §6.8-0002).
        let vk = derive_structure_key_token(cell, &ops, "vulkan:spirv1.6").expect("derives");
        assert_eq!(
            vk,
            "sk3|gem|f32|vulkan:spirv1.6|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/f32/f32/f32/st"
        );
    }

    /// A batched gem cell carries the conditionally-present `b<class>` right
    /// after `<kdiv>` (7 `/`-parts); the non-batched twin omits it entirely
    /// (6 parts). Byte-for-byte the staged `sk3_gem_batched_cell` golden.
    #[test]
    fn sk3_gem_batched_cell_golden() {
        let ops = [f32c(&[256, 4096]), f32c(&[4096, 4096]), f32c(&[256, 4096])];
        let batched = GemCell { batch: Some(256), ..gem_f32(256, 4096, 4096) };
        let token = derive_structure_key_token(
            FuelOpCategory::Contraction(batched),
            &ops,
            "cuda:sm90",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk3|gem|f32|cuda:sm90|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|cmll/d16/bm/f32/f32/f32/st"
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
    /// `sk3_simt_f32_vs_tf32_distinct_by_mp` goldens.
    #[test]
    fn sk3_gem_simt_f32_vs_tf32_distinct_by_mp() {
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
        let tf32 = derive_structure_key_token(
            FuelOpCategory::Contraction(tf32_cell),
            &ops,
            "cuda:sm90",
        )
        .expect("derives");
        assert_eq!(
            simt,
            "sk3|gem|f32|cuda:sm90|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/f32/f32/f32/st"
        );
        assert_eq!(
            tf32,
            "sk3|gem|f32|cuda:sm90|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/f32/f32/f32/rm"
        );
        assert_ne!(simt, tf32, "SIMT-f32 and TF32 must not collide");
    }

    /// The mixed-precision FP8 cell the sk3 bump exists to disambiguate,
    /// with the variant-explicit `e4m3fn` spelling in BOTH the primary and
    /// the weight coordinate. Byte-for-byte the staged
    /// `sk3_mixed_precision_fp8_disambiguated` golden (its second, fully
    /// Fuel-representable vector: E4M3×E4M3→F16, f32 acc, bit-stable; the
    /// e5m2-weight first vector is not derivable — Fuel's `DType` carries no
    /// `e5m2` storage dtype).
    #[test]
    fn sk3_gem_mixed_precision_fp8_golden() {
        // f8 operands at a 4-byte-aligned view (start_offset 4 → align 4) so
        // the 1-byte dtype derives v4 (matching the staged golden's sub-keys),
        // not the offset-0 v8.
        let f8 = |dims: &[usize]| {
            FdxOperandDesc::from_layout(
                &Layout::new(
                    Shape::from_dims(dims),
                    Layout::contiguous(Shape::from_dims(dims)).stride().iter().copied().collect::<StrideVec>(),
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
        let token = derive_structure_key_token(
            FuelOpCategory::Contraction(cell),
            &ops,
            "cuda:sm90",
        )
        .expect("derives");
        assert_eq!(
            token,
            "sk3|gem|e4m3fn|cuda:sm90|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/e4m3fn/f32/f16/st"
        );
        // The f32-out twin is a DISTINCT token (the sk2 collision resolved
        // in-key by the precision coordinates, §6.6-0018).
        let twin = GemCell { out_dtype: DType::F32, ..cell };
        let twin_token = derive_structure_key_token(
            FuelOpCategory::Contraction(twin),
            &ops,
            "cuda:sm90",
        )
        .expect("derives");
        assert_ne!(token, twin_token, "mixed-precision FP8 cells must not collide under sk3");
    }

    /// A gem cell declines (never guesses) on a precision coordinate outside
    /// the closed §6.1 set (an MX dtype) or an invalid negative role extent.
    #[test]
    fn gem_declines_unmappable_or_invalid() {
        let ops = [f32c(&[8, 4096]), f32c(&[4096, 4096]), f32c(&[8, 4096])];
        let mx_weight = GemCell { weight_dtype: DType::F4, ..gem_f32(8, 4096, 4096) };
        assert_eq!(
            derive_structure_key_token(FuelOpCategory::Contraction(mx_weight), &ops, "cuda:sm89"),
            None
        );
        let negative_m = gem_f32(-1, 4096, 4096);
        assert_eq!(
            derive_structure_key_token(FuelOpCategory::Contraction(negative_m), &ops, "cuda:sm89"),
            None
        );
        let negative_batch = GemCell { batch: Some(-2), ..gem_f32(8, 4096, 4096) };
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
            (8i64, 9, 129, "ctsm/da"),      // K=129 odd → da
            (128, 2048, 2049, "csml/da"),   // K=2049 odd → da
            (1, 129, 24, "ctms/d8"),        // K=24 → d8 (mod 16 = 8)
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
        const ALL: [DType; 15] = [
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
            DType::F6E2M3,
            DType::F6E3M2,
            DType::F4,
            DType::F8E8M0,
        ];
        let assert_clean = |token: &str| {
            assert!(!token.contains("f32s"), "retired f32s spelling in {token}");
            assert!(!token.contains("fnuz"), "reserved fnuz spelling in {token}");
            assert!(!token.contains("|e4m3|"), "retired bare e4m3 (primary) in {token}");
            assert!(!token.contains("/e4m3/"), "retired bare e4m3 (gem group) in {token}");
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
                &[co(&[8, 4096], dt), co(&[4096, 4096], dt), co(&[8, 4096], dt)],
                "cuda:sm89",
            ) {
                assert_clean(&token);
            }
        }
        // The OCP FP8 dtype spells e4m3fn, in both positions.
        let token = derive_structure_key_token(
            FuelOpCategory::BinaryElementwise,
            &[co(&[4096], DType::F8E4M3)],
            "cuda:sm89",
        )
        .expect("e4m3fn derives");
        assert!(token.starts_with("sk3|bin|e4m3fn|"), "got {token}");
    }

    // ---- typed declines ------------------------------------------------------

    #[test]
    fn declines_rather_than_guessing() {
        // Unmapped dtype (MX F4 — not in the KISS §6.1 set) → typed decline.
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
            derive_structure_key_token(
                FuelOpCategory::BinaryElementwise,
                &[broken],
                "cuda:sm89"
            ),
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
        assert_eq!(token, "sk3|red|f32|cuda:sm89|ix32|grid|r1|co/00/v1/d16/f|rall");
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
            "sk3|red|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f|x01"
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
        assert_eq!(token, "sk3|red|f32|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/00/v1/da/f|rlast");
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
        assert_eq!(token, "sk3|red|f32|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/00/v1/da/f|rall");
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
        assert_eq!(token, "sk3|red|f32|cuda:sm89|ix32|warp|r1|co/00/v1/d8/f;co/00/v1/da/f|rall");
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
            "sk3|red|f32|cuda:sm89|ix32|block|r4|co/00/v1/da/f;co/00/v1/da/f|x0a"
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
        assert_eq!(token, "sk3|bin|f32|cuda:sm89|ix32|warp|r2|co/00/v1/da/f|-");
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
        assert_eq!(token, "sk3|bin|f32|cuda:sm89|ix32|warp|r1|co/00/v1/da/f|-");
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
        let token = derive_structure_key_token(
            FuelOpCategory::BinaryElementwise,
            &[desc],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(token, "sk3|bin|f32|cuda:sm89|ix32|warp|r2|co/00/v1/da/r|-");
    }

    /// `alignment = 0` (unspecified base) cannot honor a packed load ⇒ v1
    /// (§6.5-0009; the sk2-era `0 % vbytes == 0` would have vectorized).
    #[test]
    fn alignment_zero_derives_v1() {
        let mut desc = f32c(&[4096]);
        desc.align_bytes = 0;
        let token = derive_structure_key_token(
            FuelOpCategory::BinaryElementwise,
            &[desc],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(token, "sk3|bin|f32|cuda:sm89|ix32|grid|r1|co/00/v1/d16/f|-");
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
            "sk3|bin|f32|cuda:sm89|ix32|grid|r2|br/01/v1/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-"
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
            "sk3|bin|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;br/01/v1/d16/f;co/00/v4/d16/f|-"
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
            "sk3|bin|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-"
        );
        let inplace = derive_structure_key_token(
            FuelOpCategory::BinaryElementwise,
            &[op.clone(), op],
            "cuda:sm89",
        )
        .expect("derives");
        assert_eq!(
            inplace,
            "sk3|bin|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f|-"
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
            "sk3|bin|f16|cuda:sm89|ix32|grid|r2|co/00/v8/d16/f;co/00/v8/d16/f|-"
        );
    }
}
