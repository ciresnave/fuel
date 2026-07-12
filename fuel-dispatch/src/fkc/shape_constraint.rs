//! §3.5 shape/rank constraint vocabulary — parser + probe-shape solver.
//!
//! Structured like `cost_expr.rs`. The `<expr>` grammar inside `dim[i]=<expr>`,
//! `divisible(...)`, `capacity_ge(...)` reuses `cost_expr::parse_expr`.
use crate::fkc::cost_expr::{parse_expr as parse_cost_expr, CostNode};
use crate::fkc::error::FkcError;
use crate::fkc::schema::TensorDesc;
use crate::fkc::ImportWarning;
use fuel_ir::{DType, Shape};

pub type ProbeCombo = Vec<(String, fuel_ir::Shape, fuel_ir::DType)>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RankSpec { Exact(usize), Any, Range { lo: usize, hi: Option<usize> } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisIndex { FromStart(usize), FromEnd(usize) } // dim[2]=FromStart(2); dim[-1]=FromEnd(1)

#[derive(Debug, Clone, PartialEq)]
pub enum ShapeAtom {
    SameAs(String), SameRank(String), Rank(RankSpec), BroadcastTo(String), LastDimEq(String),
    DimEq { axis: AxisIndex, expr: CostNode },
    Divisible { lhs: CostNode, rhs: CostNode },
    CapacityGe { axis: AxisIndex, sym: String },
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ShapeConstraint { pub atoms: Vec<ShapeAtom>, pub freetext: Vec<String> }

fn parse_axis(s: &str) -> Option<AxisIndex> {
    let s = s.trim();
    if let Some(n) = s.strip_prefix('-') { n.trim().parse::<usize>().ok().map(AxisIndex::FromEnd) }
    else { s.parse::<usize>().ok().map(AxisIndex::FromStart) }
}

/// `4` | `any` | `2..=4` | `2..` -> RankSpec; None on anything else.
pub fn parse_rank_spec(s: &str) -> Option<RankSpec> {
    let s = s.trim();
    if s == "any" { return Some(RankSpec::Any); }
    if let Ok(n) = s.parse::<usize>() { return Some(RankSpec::Exact(n)); }
    if let Some((lo, hi)) = s.split_once("..=") {
        return Some(RankSpec::Range { lo: lo.trim().parse().ok()?, hi: Some(hi.trim().parse().ok()?) });
    }
    if let Some(lo) = s.strip_suffix("..") {
        return Some(RankSpec::Range { lo: lo.trim().parse().ok()?, hi: None });
    }
    None
}

/// Split `a, b` on the FIRST top-level comma, tracking `(` and `[` depth so
/// `capacity_ge(dim[0], seqlen)` / `divisible(q.dim[2], k.dim[2])` split correctly.
fn split_two_args(inner: &str) -> Option<(&str, &str)> {
    let mut depth = 0i32;
    for (i, c) in inner.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => return Some((&inner[..i], &inner[i + 1..])),
            _ => {}
        }
    }
    None
}

pub fn parse_shape_constraint(raw: &str, section: &str, operand: &str)
    -> Result<ShapeConstraint, FkcError>
{
    let mut out = ShapeConstraint::default();
    for seg_raw in raw.split(';') {
        let seg = seg_raw.trim();
        if seg.is_empty() { continue; }
        let unparse = || FkcError::UnparseableShapeConstraint {
            section: section.into(), operand: operand.into(), raw: seg.to_string() };
        if let Some(r) = seg.strip_prefix("same_as=")    { out.atoms.push(ShapeAtom::SameAs(r.trim().into())); continue; }
        if let Some(r) = seg.strip_prefix("same_rank=")  { out.atoms.push(ShapeAtom::SameRank(r.trim().into())); continue; }
        if let Some(r) = seg.strip_prefix("broadcast_to="){ out.atoms.push(ShapeAtom::BroadcastTo(r.trim().into())); continue; }
        if let Some(r) = seg.strip_prefix("last_dim_eq=") { out.atoms.push(ShapeAtom::LastDimEq(r.trim().into())); continue; }
        if let Some(r) = seg.strip_prefix("rank=") {                 // COMMITTED keyword
            out.atoms.push(ShapeAtom::Rank(parse_rank_spec(r).ok_or_else(unparse)?)); continue;
        }
        if seg.starts_with("divisible(") {                          // COMMITTED: require close paren
            let inner = seg.strip_prefix("divisible(").and_then(|s| s.strip_suffix(')')).ok_or_else(unparse)?;
            let (a, b) = split_two_args(inner).ok_or_else(unparse)?;
            let lhs = parse_cost_expr(a.trim()).map_err(|_| unparse())?;
            let rhs = parse_cost_expr(b.trim()).map_err(|_| unparse())?;
            out.atoms.push(ShapeAtom::Divisible { lhs, rhs }); continue;
        }
        if seg.starts_with("capacity_ge(") {                        // COMMITTED: require close paren
            let inner = seg.strip_prefix("capacity_ge(").and_then(|s| s.strip_suffix(')')).ok_or_else(unparse)?;
            let (a, b) = split_two_args(inner).ok_or_else(unparse)?;
            let axis = a.trim().strip_prefix("dim[").and_then(|s| s.strip_suffix(']'))
                .and_then(parse_axis).ok_or_else(unparse)?;
            out.atoms.push(ShapeAtom::CapacityGe { axis, sym: b.trim().to_string() }); continue;
        }
        if seg.starts_with("dim[") {
            if let Some(close) = seg.find(']') {
                let idx = &seg["dim[".len()..close];
                let after = seg[close + 1..].trim_start();
                match (parse_axis(idx), after.strip_prefix('=')) {
                    // committed `dim[<int>]=<expr>` with a SINGLE '=' (not `==`)
                    (Some(axis), Some(rhs)) if !rhs.starts_with('=') => {
                        let rhs = rhs.trim();
                        if rhs.is_empty() { return Err(unparse()); }
                        out.atoms.push(ShapeAtom::DimEq { axis, expr: parse_cost_expr(rhs).map_err(|_| unparse())? });
                        continue;
                    }
                    // symbolic index (`dim[i]`) or `==` ⇒ pseudocode ⇒ free text
                    _ => { out.freetext.push(seg.to_string()); continue; }
                }
            }
        }
        out.freetext.push(seg.to_string()); // no recognized keyword ⇒ §3.5 notes-style free text
    }
    Ok(out)
}

#[derive(Clone, Copy)]
struct SeedProfile { base: i64, odd_last: bool }
const PROFILES: [SeedProfile; 3] = [
    SeedProfile { base: 2, odd_last: false }, // A all-2
    SeedProfile { base: 4, odd_last: true },  // B all-4, last axis ->3
    SeedProfile { base: 8, odd_last: false }, // C all-8
];

fn resolve_rank_spec_field(v: Option<&serde_yml::Value>) -> Option<RankSpec> {
    match v {
        Some(serde_yml::Value::Number(n)) => n.as_u64().map(|u| RankSpec::Exact(u as usize)),
        Some(serde_yml::Value::String(s)) => parse_rank_spec(s),
        _ => None,
    }
}
fn rank_for_probe(spec: Option<RankSpec>) -> usize {
    match spec {
        Some(RankSpec::Exact(n)) => n,
        Some(RankSpec::Range { lo, .. }) => lo,
        Some(RankSpec::Any) | None => 4, // `any`/absent default rank 4
    }
}
fn seed_axis(profile: SeedProfile, axis: usize, rank: usize) -> i64 {
    if profile.odd_last && rank > 0 && axis == rank - 1 { 3 } else { profile.base }
}
/// First declared dtype, else first `dtype_class` expansion, else F32.
fn first_probe_dtype(d: &TensorDesc) -> DType {
    if let Some(tok) = d.dtypes.first() {
        if let Ok(dt) = crate::fkc::lower::lower_dtype(tok, "", "") { return dt; }
    }
    match d.dtype_class.as_deref() {
        Some("float") => DType::BF16, Some("int") => DType::I8, Some("uint") => DType::U8,
        _ => DType::F32,
    }
}

/// Seed the probe combos for §3.5 shape-solver rank resolution + unconstrained
/// (all-free-axis) operands. Parses each operand's `shape_constraint` now so a
/// malformed-vocabulary segment hard-errors here rather than silently later —
/// Tasks 1.3/1.4 will consume `parsed` to apply the structural atoms; for now
/// this is SEED-ONLY (no atom is applied to the probe shapes yet).
pub fn solve_probe_shapes(inputs: &[TensorDesc], section: &str, warnings: &mut Vec<ImportWarning>)
    -> Result<Vec<ProbeCombo>, FkcError>
{
    // Parse each operand's constraint now so a malformed-vocabulary segment is a
    // hard error before we build any probe (Tasks 1.3/1.4 consume `parsed`).
    let mut parsed = Vec::with_capacity(inputs.len());
    for d in inputs {
        let operand = d.name.as_deref().unwrap_or("<unnamed>");
        let sc = match &d.shape_constraint {
            Some(raw) => parse_shape_constraint(raw, section, operand)?,
            None => ShapeConstraint::default(),
        };
        parsed.push(sc);
    }
    let mut combos = Vec::with_capacity(PROFILES.len());
    for profile in PROFILES {
        let mut combo: ProbeCombo = Vec::with_capacity(inputs.len());
        for d in inputs {
            let role = d.name.clone().unwrap_or_default();
            let rank = rank_for_probe(resolve_rank_spec_field(d.rank.as_ref()));
            let dims: Vec<usize> = (0..rank).map(|a| seed_axis(profile, a, rank) as usize).collect();
            combo.push((role, Shape::from_dims(&dims), first_probe_dtype(d)));
        }
        combos.push(combo);
    }
    let _ = (&parsed, warnings); // consumed by Tasks 1.3/1.4/1.5
    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fkc::cost_expr::CostNode;

    fn desc(name: &str, dtypes: &[&str], rank: Option<u64>) -> crate::fkc::schema::TensorDesc {
        crate::fkc::schema::TensorDesc {
            name: Some(name.into()), optional: false,
            dtypes: dtypes.iter().map(|s| s.to_string()).collect(),
            dtype_class: None, layout: None,
            rank: rank.map(|r| serde_yml::Value::Number(r.into())),
            shape_constraint: None, fdx: None, device: None, substrate: None,
        }
    }

    #[test]
    fn seed_unconstrained_operands_over_three_profiles() {
        use fuel_ir::Shape;
        let inputs = vec![desc("lhs", &["F32"], Some(2)), desc("rhs", &["F32"], Some(2))];
        let mut w = Vec::new();
        let combos = solve_probe_shapes(&inputs, "s", &mut w).unwrap();
        assert_eq!(combos.len(), 3);
        assert_eq!(combos[0][0].1, Shape::from_dims(&[2, 2]));  // profile A all-2
        assert_eq!(combos[1][0].1, Shape::from_dims(&[4, 3]));  // profile B all-4, last->3
        assert_eq!(combos[2][0].1, Shape::from_dims(&[8, 8]));  // profile C all-8
        assert!(w.is_empty());
    }

    #[test]
    fn rank_any_defaults_to_4_and_open_range_uses_lo() {
        let any = desc("a", &["F32"], None); // no rank ⇒ Any ⇒ 4
        assert_eq!(solve_probe_shapes(&[any], "s", &mut Vec::new()).unwrap()[0][0].1.rank(), 4);
        let mut open = desc("b", &["F32"], None);
        open.rank = Some(serde_yml::Value::String("2..".into()));
        assert_eq!(solve_probe_shapes(&[open], "s", &mut Vec::new()).unwrap()[0][0].1.rank(), 2);
    }

    #[test]
    fn parse_covers_vocab_freetext_and_rejects_malformed_vocab() {
        assert_eq!(parse_shape_constraint("same_as=k", "s", "v").unwrap().atoms,
                   vec![ShapeAtom::SameAs("k".into())]);
        assert_eq!(parse_shape_constraint("same_rank=k", "s", "v").unwrap().atoms,
                   vec![ShapeAtom::SameRank("k".into())]);
        assert_eq!(parse_shape_constraint("broadcast_to=x", "s", "v").unwrap().atoms,
                   vec![ShapeAtom::BroadcastTo("x".into())]);
        assert_eq!(parse_shape_constraint("last_dim_eq=x", "s", "v").unwrap().atoms,
                   vec![ShapeAtom::LastDimEq("x".into())]);
        assert_eq!(parse_shape_constraint("rank=4", "s", "x").unwrap().atoms,
                   vec![ShapeAtom::Rank(RankSpec::Exact(4))]);
        assert_eq!(parse_shape_constraint("rank=2..=4", "s", "x").unwrap().atoms,
                   vec![ShapeAtom::Rank(RankSpec::Range { lo: 2, hi: Some(4) })]);
        // `Any` and open-ended `Range{hi:None}` (parse_rank_spec branches otherwise unpinned)
        assert_eq!(parse_shape_constraint("rank=any", "s", "x").unwrap().atoms,
                   vec![ShapeAtom::Rank(RankSpec::Any)]);
        assert_eq!(parse_shape_constraint("rank=2..", "s", "x").unwrap().atoms,
                   vec![ShapeAtom::Rank(RankSpec::Range { lo: 2, hi: None })]);
        // negative axis + bare-symbol RHS (linear-quant.fkc.md:108)
        let a = parse_shape_constraint("dim[-1]=k; same_rank=b", "linear", "a").unwrap();
        assert_eq!(a.atoms.len(), 2);
        match &a.atoms[0] {
            ShapeAtom::DimEq { axis, expr } => {
                assert_eq!(*axis, AxisIndex::FromEnd(1));
                assert_eq!(*expr, CostNode::Sym("k".into()));
            }
            other => panic!("got {other:?}"),
        }
        assert_eq!(a.atoms[1], ShapeAtom::SameRank("b".into()));
        assert!(matches!(parse_shape_constraint("divisible(q.dim[2], k.dim[2])", "f", "k")
            .unwrap().atoms[0], ShapeAtom::Divisible { .. }));
        assert!(matches!(parse_shape_constraint("capacity_ge(dim[0], seqlen)", "f", "kv")
            .unwrap().atoms[0], ShapeAtom::CapacityGe { .. }));
        // bracket-depth-aware split_two_args: the outer `divisible(...)` comma
        // must split at DEPTH 0, not at the nested comma inside `foo(a, b)`. A
        // naive `inner.split_once(',')` would split lhs="foo(a" / rhs=" b), c" —
        // "foo(a" fails to parse (unbalanced '(') and the whole call errors.
        let nested = parse_shape_constraint("divisible(foo(a, b), c)", "f", "k").unwrap();
        assert_eq!(nested.atoms, vec![ShapeAtom::Divisible {
            lhs: CostNode::Call {
                name: "foo".into(),
                args: vec![CostNode::Sym("a".into()), CostNode::Sym("b".into())],
            },
            rhs: CostNode::Sym("c".into()),
        }]);
        // free text: valid-vocab head + prose tail (shape-ops.fkc.md:639) — NOT rejected
        let mixed = parse_shape_constraint(
            "same_as=out; read-modify-written in place (this operand IS the output)",
            "shape-ops", "dst").unwrap();
        assert_eq!(mixed.atoms, vec![ShapeAtom::SameAs("out".into())]);
        assert_eq!(mixed.freetext.len(), 1);
        // pure free text (shape-ops.fkc.md:721)
        let ft = parse_shape_constraint("byte length >= 4 (one u32)", "shape-ops", "seed").unwrap();
        assert!(ft.atoms.is_empty());
        assert_eq!(ft.freetext.len(), 1);
        // symbolic index + `==` (shape-ops.fkc.md:98) ⇒ free text, not reject
        let sym_i = parse_shape_constraint("dim[i] == in_shape[i]", "shape-ops", "out").unwrap();
        assert!(sym_i.atoms.is_empty());
        assert_eq!(sym_i.freetext.len(), 1);
        // HARD reject: keyword-committed segment with malformed argument
        assert!(matches!(parse_shape_constraint("rank=banana", "s", "x").unwrap_err(),
                         FkcError::UnparseableShapeConstraint { .. }));
        assert!(parse_shape_constraint("divisible(x.dim[0]", "s", "x").is_err()); // unclosed (
        assert!(parse_shape_constraint("dim[0]=", "s", "x").is_err());           // committed, empty rhs
    }
}
