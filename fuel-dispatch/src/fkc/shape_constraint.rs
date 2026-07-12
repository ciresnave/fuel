//! §3.5 shape/rank constraint vocabulary — parser + probe-shape solver.
//!
//! Structured like `cost_expr.rs`. The `<expr>` grammar inside `dim[i]=<expr>`,
//! `divisible(...)`, `capacity_ge(...)` reuses `cost_expr::parse_expr`.
use std::collections::HashMap;

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

/// Per-profile solver state: each operand's working dims (seeded, then mutated
/// in place by `apply_atom`), a lazily-seeded shared-symbol env (so a bare
/// `dim[i]=k` on one operand and `dim[j]=k` on another bind the SAME value),
/// and the profile's seed base (the value a freshly-seen symbol takes).
struct Solve { dims: HashMap<String, Vec<i64>>, sym: HashMap<String, i64>, base: i64 }

/// Recognize `dim[i]` (self) or `role.dim[i]` as a dim reference.
fn as_dim_ref(node: &CostNode) -> Option<(Option<String>, AxisIndex)> {
    if let CostNode::Index { base, index } = node {
        let axis = match &**index {
            CostNode::Lit(v) => AxisIndex::FromStart(*v as usize),
            CostNode::Neg(inner) => if let CostNode::Lit(v) = &**inner { AxisIndex::FromEnd(*v as usize) } else { return None },
            _ => return None,
        };
        if let CostNode::Sym(s) = &**base {
            return Some(if let Some(r) = s.strip_suffix(".dim") { (Some(r.to_string()), axis) }
                        else if s == "dim" { (None, axis) } else { return None });
        }
    }
    None
}

fn axis_to_index(axis: AxisIndex, rank: usize) -> Option<usize> {
    match axis { AxisIndex::FromStart(i) => Some(i), AxisIndex::FromEnd(n) => rank.checked_sub(n) }
}

/// Evaluate a CostNode to a concrete i64. None ⇒ genuinely unresolvable.
fn eval_dim_expr(node: &CostNode, s: &mut Solve, ranks: &HashMap<String, usize>, self_role: &str) -> Option<i64> {
    use crate::fkc::cost_expr::BinOp::*;
    match node {
        CostNode::Lit(v) => Some(*v as i64),
        CostNode::Neg(i) => eval_dim_expr(i, s, ranks, self_role)?.checked_neg(),
        CostNode::Bin { op, lhs, rhs } => {
            let (l, r) = (eval_dim_expr(lhs, s, ranks, self_role)?, eval_dim_expr(rhs, s, ranks, self_role)?);
            // Checked: `overflow-checks = true` under `cargo test` PANICS on
            // unchecked i64 overflow. `?` propagates `None` (genuinely
            // unresolvable) rather than crashing on a huge-but-syntactically-
            // valid dim expression (review Finding 1). `checked_div`/
            // `checked_rem` ALSO subsume the zero-divisor guard (both return
            // `None` for `r == 0`) AND the `i64::MIN / -1` / `i64::MIN % -1`
            // edge case, which panics unconditionally in ALL profiles
            // (release included) — that panic is hard-baked into `/`/`%`,
            // not gated by `overflow-checks` (fix-pass-2 residual finding).
            Some(match op {
                Add => l.checked_add(r)?,
                Sub => l.checked_sub(r)?,
                Mul => l.checked_mul(r)?,
                Div => l.checked_div(r)?,
                Rem => l.checked_rem(r)?,
                _ => return None,
            })
        }
        CostNode::Index { .. } => {
            let (role, axis) = as_dim_ref(node)?;
            let rrole = role.as_deref().unwrap_or(self_role);
            let idx = axis_to_index(axis, *ranks.get(rrole)?)?;
            s.dims.get(rrole)?.get(idx).copied()
        }
        CostNode::Sym(name) => Some(*s.sym.entry(name.clone()).or_insert(s.base)), // lazy-seed shared symbol
        CostNode::Call { .. } => None,
    }
}

fn warn(section: &str, message: String) -> ImportWarning { ImportWarning { section: section.into(), message } }

fn set_axis(s: &mut Solve, role: &str, axis: AxisIndex, rank: usize, v: i64) {
    if let Some(idx) = axis_to_index(axis, rank) {
        if let Some(d) = s.dims.get_mut(role) { if idx < d.len() { d[idx] = v.max(1); } }
    }
}

fn apply_atom(atom: &ShapeAtom, self_role: &str, s: &mut Solve, ranks: &HashMap<String, usize>,
              w: &mut Vec<ImportWarning>, section: &str) {
    let self_rank = *ranks.get(self_role).unwrap_or(&0);
    match atom {
        ShapeAtom::Rank(_) | ShapeAtom::SameRank(_) | ShapeAtom::CapacityGe { .. } => {} // rank-phase / trivial
        ShapeAtom::SameAs(src) | ShapeAtom::BroadcastTo(src) => match s.dims.get(src).cloned() {
            Some(src_dims) => {
                let n = self_rank.min(src_dims.len());
                if let Some(d) = s.dims.get_mut(self_role) { for a in 0..n { d[a] = src_dims[a]; } }
            }
            None => w.push(warn(section, format!("operand `{self_role}` references unknown role `{src}`; using seed shape"))),
        },
        ShapeAtom::LastDimEq(src) => match s.dims.get(src).and_then(|d| d.last().copied()) {
            Some(sr) => set_axis(s, self_role, AxisIndex::FromEnd(1), self_rank, sr),
            None => w.push(warn(section, format!("operand `{self_role}` last_dim_eq references unknown role `{src}`; using seed"))),
        },
        ShapeAtom::DimEq { axis, expr } => match eval_dim_expr(expr, s, ranks, self_role) {
            Some(v) => set_axis(s, self_role, *axis, self_rank, v),
            None => w.push(warn(section, format!("operand `{self_role}` dim rule unresolved; using seed"))),
        },
        ShapeAtom::Divisible { lhs, rhs } => {
            if let (Some((role, axis)), Some(v)) = (as_dim_ref(lhs), eval_dim_expr(rhs, s, ranks, self_role)) {
                if v > 0 {
                    let target = role.as_deref().unwrap_or(self_role).to_string();
                    let trank = *ranks.get(&target).unwrap_or(&0);
                    if let Some(idx) = axis_to_index(axis, trank) {
                        if let Some(cur) = s.dims.get(&target).and_then(|d| d.get(idx).copied()) {
                            // Checked ceil-round: `cur + (v-1)` then `* v` can each
                            // overflow i64 on an adversarial-but-parseable input;
                            // SKIP the round (leave the axis unrounded) rather than
                            // panic under `overflow-checks = true` (Finding 1).
                            if let Some(rounded) = cur.checked_add(v - 1).map(|x| x / v).and_then(|q| q.checked_mul(v)) {
                                set_axis(s, &target, axis, trank, rounded);
                            }
                        }
                    }
                }
            } else if let CostNode::Sym(k) = lhs {
                if let Some(v) = eval_dim_expr(rhs, s, ranks, self_role) {
                    if v > 0 {
                        let e = s.sym.entry(k.clone()).or_insert(s.base);
                        if let Some(rounded) = e.checked_add(v - 1).map(|x| x / v).and_then(|q| q.checked_mul(v)) {
                            *e = rounded;
                        }
                    }
                }
            }
        }
    }
}

/// Seed the probe combos for §3.5 shape-solver rank resolution, then APPLY the
/// structural atoms (Task 1.3): `same_as`/`broadcast_to` copy a source
/// operand's (already-solved) dims, `dim[i]=<expr>` evaluates an expression
/// (reusing `cost_expr::CostNode`, including a bare shared symbol like
/// matmul's `k`), and `divisible(dim[i], v)` rounds an axis UP to a multiple
/// of `v`. Atoms are applied in SOURCE order (Task 1.4 adds dependency/topo
/// ordering — deliberately not here). Parses each operand's `shape_constraint`
/// up front so a malformed-vocabulary segment hard-errors before any probe is
/// built.
pub fn solve_probe_shapes(inputs: &[TensorDesc], section: &str, warnings: &mut Vec<ImportWarning>)
    -> Result<Vec<ProbeCombo>, FkcError>
{
    // Parse each operand's constraint + compute its rank once (shared across
    // all profiles — the rank spec doesn't vary by profile).
    //
    // `Solve.dims`/`ranks` are keyed by a per-operand SOLVE-KEY, not the bare
    // display role: two operands sharing a role string (most plausibly two
    // UNNAMED operands, both `name.unwrap_or_default() == ""`) would
    // otherwise collide — the second insert silently overwrites the first,
    // and both later read back the same (wrong) entry (review Finding 3).
    // A named operand's key IS its name (so `same_as=k` / `role.dim[i]`
    // still resolve against named-operand keys exactly as before); an
    // unnamed operand gets a unique `#unnamed{i}` key that no `same_as=`
    // can reference — correct, since an unnamed operand can't be
    // referenced by name. The DISPLAY role in the returned `ProbeCombo`
    // stays the plain (possibly empty) name, unchanged.
    let mut parsed = Vec::with_capacity(inputs.len());
    let mut roles = Vec::with_capacity(inputs.len());
    let mut keys = Vec::with_capacity(inputs.len());
    let mut ranks: HashMap<String, usize> = HashMap::with_capacity(inputs.len());
    for (i, d) in inputs.iter().enumerate() {
        let operand = d.name.as_deref().unwrap_or("<unnamed>");
        let sc = match &d.shape_constraint {
            Some(raw) => parse_shape_constraint(raw, section, operand)?,
            None => ShapeConstraint::default(),
        };
        let role = d.name.clone().unwrap_or_default();
        let key = d.name.clone().filter(|n| !n.is_empty()).unwrap_or_else(|| format!("#unnamed{i}"));
        let rank = rank_for_probe(resolve_rank_spec_field(d.rank.as_ref()));
        ranks.insert(key.clone(), rank);
        roles.push(role);
        keys.push(key);
        parsed.push(sc);
    }

    let mut combos = Vec::with_capacity(PROFILES.len());
    for profile in PROFILES {
        // 1. Seed every operand's dims into `s.dims`, keyed by its unique SOLVE-KEY.
        let mut s = Solve { dims: HashMap::with_capacity(inputs.len()), sym: HashMap::new(), base: profile.base };
        for key in &keys {
            let rank = *ranks.get(key).unwrap_or(&0);
            let dims: Vec<i64> = (0..rank).map(|a| seed_axis(profile, a, rank)).collect();
            s.dims.insert(key.clone(), dims);
        }
        // 2. Apply every operand's atoms, in source order (Task 1.4: topo order).
        for (i, key) in keys.iter().enumerate() {
            for atom in &parsed[i].atoms {
                apply_atom(atom, key, &mut s, &ranks, warnings, section);
            }
        }
        // 3. Read the solved dims back into Shape, paired with the operand's
        //    dtype. Lookup goes through the SOLVE-KEY; the emitted tuple's
        //    role is the plain DISPLAY name.
        let mut combo: ProbeCombo = Vec::with_capacity(inputs.len());
        for ((d, role), key) in inputs.iter().zip(roles.iter()).zip(keys.iter()) {
            let dims: Vec<usize> = s.dims.get(key).map(|v| v.iter().map(|&x| x.max(1) as usize).collect()).unwrap_or_default();
            combo.push((role.clone(), Shape::from_dims(&dims), first_probe_dtype(d)));
        }
        combos.push(combo);
    }
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

    #[test]
    fn solve_same_as_copies_dims_and_divisible_rounds_up() {
        use fuel_ir::Shape;
        let mut k = desc("k", &["F32"], Some(3));
        k.shape_constraint = Some("divisible(dim[0], 8)".into());
        let mut v = desc("v", &["F32"], Some(3));
        v.shape_constraint = Some("same_as=k".into());
        let combos = solve_probe_shapes(&[k, v], "s", &mut Vec::new()).unwrap();
        let a = &combos[0]; // profile A base 2 ⇒ ceil(2/8)*8 = 8
        let ks = &a.iter().find(|(r, _, _)| r == "k").unwrap().1;
        let vs = &a.iter().find(|(r, _, _)| r == "v").unwrap().1;
        assert_eq!(ks, &Shape::from_dims(&[8, 2, 2]));
        assert_eq!(vs, ks);
    }

    // Replaces `dim_eq_bare_symbol_is_shared_across_operands` (review Finding
    // 2): that test only ever read the SAME seeded value (profile A seeds
    // every axis of a rank-2 operand to `2`), so it passed identically
    // whether or not the `sym` env was actually shared — it never exercised
    // sharing. This version MUTATES the shared symbol on operand `a`
    // (`divisible(k, 8)`, bare-symbol lhs, base 2 -> rounds to 8) and reads
    // it back on operand `b` (`dim[-1]=k`); it only passes if `b` observes
    // the write `a` made, i.e. if `Solve.sym` is truly one shared `&mut`
    // table across all operands within a profile, not a fresh env per
    // operand.
    #[test]
    fn bare_symbol_is_shared_mutably_across_operands() {
        // operand a bumps the shared symbol k (2 -> 8 via divisible-on-symbol);
        // operand b then READS k via dim[-1]=k. b sees 8 ONLY if the sym env is
        // shared. With independent per-operand sym envs, b would seed its own k=2.
        let mut a = desc("a", &["F32"], Some(2));
        a.shape_constraint = Some("divisible(k, 8)".into()); // bumps shared symbol k
        let mut b = desc("b", &["F32"], Some(2));
        b.shape_constraint = Some("dim[-1]=k".into());       // b.last = shared k
        let combos = solve_probe_shapes(&[a, b], "s", &mut Vec::new()).unwrap();
        let a0 = &combos[0]; // profile A, base 2
        let bk = a0.iter().find(|(r, _, _)| r == "b").unwrap().1.dims()[1];
        assert_eq!(bk, 8, "b's last axis must observe the SHARED, bumped k (=8), not a fresh k=2");
    }

    /// Review Finding 1: unchecked i64 arithmetic must not panic under
    /// `overflow-checks = true` (the default `cargo test` profile) on a
    /// syntactically-valid-but-huge `dim[i]=<expr>`. The multiply overflows
    /// i64 (10^6 * 10^6 * 10^6 * 10^6 = 10^24 >> i64::MAX) so `eval_dim_expr`
    /// must return `None` (checked op propagates via `?`), degrading to a
    /// warning + the seed shape rather than crashing.
    #[test]
    fn dim_expr_overflow_degrades_without_panic() {
        let mut a = desc("a", &["F32"], Some(2));
        a.shape_constraint = Some("dim[0]=1000000*1000000*1000000*1000000".into());
        let mut w = Vec::new();
        let result = solve_probe_shapes(&[a], "s", &mut w);
        assert!(result.is_ok(), "overflow must degrade gracefully, not panic or error");
        assert!(!w.is_empty(), "the unresolved (overflowed) atom should surface an ImportWarning");
    }

    /// Fix-pass-2 residual: `i64::MIN / -1` (and `% -1`) panics UNCONDITIONALLY
    /// — in every cargo profile, release included — because that trap is
    /// hard-baked into the CPU `idiv`/`div` instruction, not gated by
    /// `overflow-checks`. `-i64::MIN` additionally panics under
    /// `overflow-checks = true` (the default `cargo test` profile). Both must
    /// degrade to a warning + the untouched seed shape via `checked_div` /
    /// `checked_neg` rather than crash the FKC importer at shape-solve time.
    ///
    /// The literal `9223372036854775807` (i64::MAX, 19 digits) round-trips
    /// through `CostNode::Lit(f64)` as `9223372036854775808.0` (nearest
    /// representable `f64`, = 2^63, since doubles can't represent all
    /// 19-digit integers exactly) and `as i64`-saturates back down to
    /// `i64::MAX` — so `-9223372036854775807 - 1` evaluates to EXACTLY
    /// `i64::MIN` (`checked_sub` sees `-i64::MAX - 1`, which fits). That
    /// lands the div/rem and neg cases squarely on the target operand:
    /// `dim[0]=(i64::MIN)/-1` drives the `Div` arm with `l=i64::MIN, r=-1`;
    /// `dim[0]=-(i64::MIN)` drives the `Neg` arm with `x=i64::MIN`.
    #[test]
    fn pathological_int_min_div_and_neg_degrade_without_panic() {
        use fuel_ir::Shape;
        let mut a = desc("a", &["F32"], Some(1));
        a.shape_constraint = Some("dim[0]=(-9223372036854775807-1)/-1".into()); // i64::MIN / -1
        let mut wa = Vec::new();
        let ra = solve_probe_shapes(&[a], "s", &mut wa);
        assert!(ra.is_ok(), "MIN/-1 must degrade, not panic");
        assert!(!wa.is_empty(), "the unresolved (MIN/-1) atom should surface an ImportWarning");
        // Degraded ⇒ set_axis was never called ⇒ profile A's rank-1 operand
        // keeps its seeded shape (base 2; odd_last doesn't apply to profile A).
        assert_eq!(ra.unwrap()[0][0].1, Shape::from_dims(&[2]));

        let mut b = desc("b", &["F32"], Some(1));
        b.shape_constraint = Some("dim[0]=-(-9223372036854775807-1)".into()); // -(i64::MIN)
        let mut wb = Vec::new();
        let rb = solve_probe_shapes(&[b], "s", &mut wb);
        assert!(rb.is_ok(), "-(i64::MIN) must degrade, not panic");
        assert!(!wb.is_empty(), "the unresolved (Neg MIN) atom should surface an ImportWarning");
        assert_eq!(rb.unwrap()[0][0].1, Shape::from_dims(&[2]));
    }

    /// Review Finding 3: `Solve.dims`/`ranks` must be keyed by a unique
    /// per-operand SOLVE-KEY, not the bare (possibly-empty) role string —
    /// otherwise two unnamed operands (`name.unwrap_or_default() == ""` for
    /// both) collide on the same HashMap entry and the second silently
    /// overwrites the first. Under the pre-fix role-"" keying, BOTH the
    /// `ranks[""]` entry and the `s.dims[""]` entry are only ever the
    /// LAST-inserted unnamed operand's — so operand 0 would incorrectly
    /// read back operand 1's rank-3 shape (first assert below fails: actual
    /// rank 3, expected 2). With a unique `#unnamed{i}` key per operand,
    /// each keeps its own rank/dims.
    #[test]
    fn two_unnamed_operands_are_not_aliased() {
        // Two unnamed operands with DIFFERENT ranks must get their own shapes,
        // not alias to the last-inserted rank.
        let a = crate::fkc::schema::TensorDesc { name: None, optional: false, dtypes: vec!["F32".into()], dtype_class: None, layout: None, rank: Some(serde_yml::Value::Number(2u64.into())), shape_constraint: None, fdx: None, device: None, substrate: None };
        let mut b = a.clone();
        b.rank = Some(serde_yml::Value::Number(3u64.into()));
        let combos = solve_probe_shapes(&[a, b], "s", &mut Vec::new()).unwrap();
        assert_eq!(combos[0][0].1.rank(), 2, "first unnamed operand keeps rank 2");
        assert_eq!(combos[0][1].1.rank(), 3, "second unnamed operand keeps rank 3, not aliased to 2");
    }
}
