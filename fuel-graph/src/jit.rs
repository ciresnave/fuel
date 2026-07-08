//! Kernel-seam JIT — the Fuel-side projection + structural matcher
//! (kernel-seam-interop §5; fkc-fusion-patterns §3/§3a). The frozen grammar
//! types ([`OpTag`], [`OpAttrs`], [`PatternNode`]) live in
//! [`fuel_kernel_seam_types`] and are re-exported here; this module holds the
//! two parts that need the graph: the `Op -> OpTag` projection [`op_to_tag`]
//! and the structural matcher [`match_region`] (the declarative-fusion engine
//! behind `PatternKind::Declarative`).
//!
//! One [`PatternNode`] serves the **JIT region** (Fuel -> synthesizer, "build a
//! kernel for this subgraph"), a contract's `pattern:` **re-fuse rule**, and a
//! synthesized op's **`decompose`** (the region re-emitted).

use crate::{Graph, NodeId, Op};
use std::collections::BTreeMap;

pub use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};

// ===========================================================================
// op_to_tag — the Op -> OpTag projection (stays Fuel-side: it needs the graph Op)
// ===========================================================================

/// Project a graph [`Op`] to its functional [`OpTag`]. Returns `None` for
/// in-place variants, structural/bookkeeping ops, and `Op::Fused` (a fused op
/// isn't a region node — its *decomposition* is). A `None` op in a region is an
/// honest "outside the vocabulary" miss, never a crash.
pub fn op_to_tag(op: &Op) -> Option<OpTag> {
    Some(match op {
        Op::Add => OpTag::Add,
        Op::Sub => OpTag::Sub,
        Op::Mul => OpTag::Mul,
        Op::Div => OpTag::Div,
        Op::Maximum => OpTag::Maximum,
        Op::Minimum => OpTag::Minimum,
        Op::Pow => OpTag::Pow,
        Op::Rem => OpTag::Rem,
        Op::Neg => OpTag::Neg,
        Op::Abs => OpTag::Abs,
        Op::Sqr => OpTag::Sqr,
        Op::Sqrt => OpTag::Sqrt,
        Op::Rsqrt => OpTag::Rsqrt,
        Op::Recip => OpTag::Recip,
        Op::Exp => OpTag::Exp,
        Op::Log => OpTag::Log,
        Op::Sin => OpTag::Sin,
        Op::Cos => OpTag::Cos,
        Op::Tanh => OpTag::Tanh,
        Op::Sigmoid => OpTag::Sigmoid,
        Op::Silu => OpTag::Silu,
        Op::Gelu => OpTag::Gelu,
        Op::GeluErf => OpTag::GeluErf,
        Op::Relu => OpTag::Relu,
        Op::Erf => OpTag::Erf,
        Op::Step => OpTag::Step,
        Op::Floor => OpTag::Floor,
        Op::Ceil => OpTag::Ceil,
        Op::Round => OpTag::Round,
        Op::Sign => OpTag::Sign,
        Op::AddScalar(_) => OpTag::AddScalar,
        Op::MulScalar(_) => OpTag::MulScalar,
        Op::PowI(_) => OpTag::PowI,
        Op::Clamp { .. } => OpTag::Clamp,
        Op::Equal => OpTag::Equal,
        Op::Ne => OpTag::Ne,
        Op::Lt => OpTag::Lt,
        Op::Le => OpTag::Le,
        Op::Gt => OpTag::Gt,
        Op::Ge => OpTag::Ge,
        Op::Where => OpTag::Where,
        Op::MaskedFill { .. } => OpTag::MaskedFill,
        Op::SumAll => OpTag::SumAll,
        Op::MaxAll => OpTag::MaxAll,
        Op::MinAll => OpTag::MinAll,
        Op::MeanAll => OpTag::MeanAll,
        Op::SumDim(_) => OpTag::SumDim,
        Op::MeanDim(_) => OpTag::MeanDim,
        Op::ReduceSumTo(_) => OpTag::ReduceSumTo,
        Op::ReduceMaxTo(_) => OpTag::ReduceMaxTo,
        Op::CumSum { .. } => OpTag::CumSum,
        Op::MatMul => OpTag::MatMul,
        Op::Transpose => OpTag::Transpose,
        Op::Permute(_) => OpTag::Permute,
        Op::Reshape(_) => OpTag::Reshape,
        Op::BroadcastTo(_) => OpTag::BroadcastTo,
        Op::Unsqueeze { .. } => OpTag::Unsqueeze,
        Op::Squeeze { .. } => OpTag::Squeeze,
        Op::Cast(_) => OpTag::Cast,
        Op::Slice { .. } => OpTag::Slice,
        Op::Concat { .. } => OpTag::Concat,
        Op::Flip { .. } => OpTag::Flip,
        Op::Roll { .. } => OpTag::Roll,
        Op::Pad { .. } => OpTag::Pad,
        Op::Triu { .. } => OpTag::Triu,
        Op::Tril { .. } => OpTag::Tril,
        Op::IndexSelect { .. } => OpTag::IndexSelect,
        Op::Gather { .. } => OpTag::Gather,
        Op::IndexAdd { .. } => OpTag::IndexAdd,
        Op::ScatterAdd { .. } => OpTag::ScatterAdd,
        Op::LogSoftmaxLastDim => OpTag::LogSoftmaxLastDim,
        Op::Iota { .. } => OpTag::Iota,
        // In-place variants, structural / bookkeeping ops, and Op::Fused are
        // not region nodes.
        _ => return None,
    })
}

// ===========================================================================
// match_region — the structural declarative matcher (§3a)
// ===========================================================================

/// The §4.1 commutative ops, whose operands match order-independently (§3a.2a).
fn is_commutative(op: OpTag) -> bool {
    matches!(op, OpTag::Add | OpTag::Mul | OpTag::Maximum | OpTag::Minimum)
}

/// Project a graph [`Op`]'s load-bearing non-tensor parameters into an
/// [`OpAttrs`] — the graph-side mirror of the region-side `attrs` a
/// [`PatternNode::Op`] carries. The graph node stores these as *typed* `Op`
/// payloads (`Op::Permute(Vec<usize>)`, `Op::AddScalar(f64)`, `Op::SumDim(usize)`,
/// `Op::BroadcastTo(Shape)`, …); this reads them out into the flat `OpAttrs`
/// surface so [`attrs_match`] can compare a pattern's `attrs` against them
/// without the seam-types crate needing to know about the graph `Op` (it stays
/// dependency-free). Ops with no attr payload project to `OpAttrs::default()`
/// (all fields empty), which the wildcard rule treats as "no constraint".
fn op_to_attrs(op: &Op) -> OpAttrs {
    let mut a = OpAttrs::default();
    match op {
        // Scalar-param ops → `scalars` (the region's slot snapshot; F1).
        Op::AddScalar(v) | Op::MulScalar(v) => a.scalars = vec![*v],
        Op::Clamp { min, max } => a.scalars = vec![*min, *max],
        // Dim-bearing ops → `axis`.
        Op::SumDim(d) | Op::MeanDim(d) => a.axis = Some(*d as i64),
        Op::Triu { diagonal } | Op::Tril { diagonal } => a.axis = Some(*diagonal),
        // Permute/Transpose → absolute `perm` (F1/F2a). `Transpose` is the
        // rank-2 last-two-axes special case; without the input rank on the op
        // itself it projects to an empty perm (a wildcard) here — a `Permute`
        // pattern is the discriminating form.
        Op::Permute(axes) => a.perm = axes.iter().map(|&x| x as u8).collect(),
        // Shape-target ops → `target_shape` (BroadcastTo + Reshape share it; F1).
        Op::BroadcastTo(shape) | Op::Reshape(shape) => {
            a.target_shape = shape.dims().iter().map(|&d| d as i64).collect()
        }
        // Squeeze/Unsqueeze → single-element `dims` (F1).
        Op::Unsqueeze { dim } | Op::Squeeze { dim } => a.dims = vec![*dim as u8],
        _ => {}
    }
    a
}

/// Compare a pattern node's `attrs` against the graph node's projected attrs
/// with **wildcard-on-unset** semantics: an *empty/unset* field on the pattern
/// is a wildcard (matches any graph value); a *set* field must equal the graph
/// node's value. This is what keeps every existing attr-agnostic pattern (all
/// authored with `OpAttrs::default()`) matching after attrs become comparable,
/// while letting a layout/scalar pattern that *sets* a field discriminate.
///
/// `Vec` fields are unset ⇔ empty; `axis: Option` is unset ⇔ `None`. A set
/// pattern field must equal the graph projection exactly (absolute perm, F2a).
fn attrs_match(pattern: &OpAttrs, node: &OpAttrs) -> bool {
    (pattern.scalars.is_empty() || pattern.scalars == node.scalars)
        && (pattern.axis.is_none() || pattern.axis == node.axis)
        && (pattern.perm.is_empty() || pattern.perm == node.perm)
        && (pattern.target_shape.is_empty() || pattern.target_shape == node.target_shape)
        && (pattern.dims.is_empty() || pattern.dims == node.dims)
}

/// Match a declarative region [`PatternNode`] against the graph rooted at
/// `root` (the subgraph **sink**, §3a.1). Returns the region's external inputs
/// in `bind`-index order on a match, or `None`. This is the structural core of
/// the declarative matcher (`PatternKind::Declarative`); the §5 `guard:`/`extract:`
/// layers and the `see_through`-set wrappers compose on top.
///
/// Implements: positional exact tensor-arity (scalar params are attributes, not
/// operands — §3a.2); commutative-op order-independence (§3a.2a, by trying both
/// orderings); the **interior sole-consumer guard** (§3a.4 — a matched Op that is
/// neither the root nor a `bind` leaf must feed *only* the fusion, else fusing
/// duplicates its computation and we decline); and the repeated-`bind`
/// node-identity guard (§3.2). `consumers(n)` returns node `n`'s consumer count.
pub fn match_region(
    graph: &Graph,
    root: NodeId,
    pattern: &PatternNode,
    consumers: &dyn Fn(NodeId) -> usize,
) -> Option<Vec<NodeId>> {
    match_region_extract(graph, root, pattern, consumers).map(|(binds, _)| binds)
}

/// [`match_region`] plus the **`extract:` layer** (FKC §5.3): alongside the
/// bound inputs, return the matched region's live scalar values — one entry per
/// scalar a **slot** pattern node left open (a scalar-carrying op whose pattern
/// `attrs.scalars` is empty/wildcard), read from the matched graph node, in
/// **pattern pre-order** (the canonical slot order the recipe's re-emit and a
/// synthesized kernel's trailing `p{i}` launch args both use). A pattern node
/// with *baked* scalars is a constant of the pattern (the attr guard enforced
/// equality), not a slot, and extracts nothing. A slotless region extracts `[]`.
pub fn match_region_extract(
    graph: &Graph,
    root: NodeId,
    pattern: &PatternNode,
    consumers: &dyn Fn(NodeId) -> usize,
) -> Option<(Vec<NodeId>, Vec<f64>)> {
    let mut binds: BTreeMap<u8, NodeId> = BTreeMap::new();
    let mut scalars: Vec<f64> = Vec::new();
    match_node(graph, root, pattern, true, consumers, &mut binds, &mut scalars)?;
    // Bind indices must form a contiguous [0, n) — exactly the region's inputs.
    let n = binds.len() as u8;
    if (0..n).all(|i| binds.contains_key(&i)) {
        Some(((0..n).map(|i| binds[&i]).collect(), scalars))
    } else {
        None
    }
}

fn match_node(
    graph: &Graph,
    node_id: NodeId,
    pattern: &PatternNode,
    is_root: bool,
    consumers: &dyn Fn(NodeId) -> usize,
    binds: &mut BTreeMap<u8, NodeId>,
    scalars: &mut Vec<f64>,
) -> Option<()> {
    match pattern {
        PatternNode::Any => Some(()),
        PatternNode::Bind { index } => {
            // Node-identity guard: a repeated index must bind the SAME node.
            match binds.get(index) {
                Some(&existing) if existing != node_id => None,
                _ => {
                    binds.insert(*index, node_id);
                    Some(())
                }
            }
        }
        PatternNode::SeeThrough { then } => {
            // The see_through-set skip is a follow-up; for now match `then`
            // against this node directly.
            match_node(graph, node_id, then, is_root, consumers, binds, scalars)
        }
        PatternNode::Op { op, operands, attrs } => {
            let node = graph.node(node_id);
            if op_to_tag(&node.op) != Some(*op) {
                return None;
            }
            // Attr guard (F1): a SET pattern attr must equal the graph node's
            // projected value; an empty/unset pattern attr is a wildcard, so
            // existing attr-agnostic patterns (all `OpAttrs::default()`) keep
            // matching. Op-tag is checked first, so the projection is meaningful.
            let node_attrs = op_to_attrs(&node.op);
            if !attrs_match(attrs, &node_attrs) {
                return None;
            }
            // Interior nodes (not the root, not a bind leaf) must be sole-consumer.
            if !is_root && consumers(node_id) != 1 {
                return None;
            }
            // Exact tensor-input arity (scalar/attribute params are not operands).
            let inputs = &node.inputs;
            if inputs.len() != operands.len() {
                return None;
            }
            // The `extract:` layer (§5.3): empty pattern-scalars on a
            // scalar-carrying op is a SLOT — record the live values from the
            // matched node, pre-order (before descending into operands).
            if attrs.scalars.is_empty() && !node_attrs.scalars.is_empty() {
                scalars.extend_from_slice(&node_attrs.scalars);
            }
            if is_commutative(*op) && operands.len() == 2 {
                // Try both orderings; commit the first that fully matches.
                // Clone-commit covers `scalars` too: a failed first ordering
                // must not leave partial extractions behind.
                for (a, b) in [(0usize, 1usize), (1, 0)] {
                    let mut trial = binds.clone();
                    let mut trial_scalars = scalars.clone();
                    if match_node(
                        graph,
                        inputs[a],
                        &operands[0],
                        false,
                        consumers,
                        &mut trial,
                        &mut trial_scalars,
                    )
                    .is_some()
                        && match_node(
                            graph,
                            inputs[b],
                            &operands[1],
                            false,
                            consumers,
                            &mut trial,
                            &mut trial_scalars,
                        )
                        .is_some()
                    {
                        *binds = trial;
                        *scalars = trial_scalars;
                        return Some(());
                    }
                }
                return None;
            }
            for (child_pat, &child_id) in operands.iter().zip(inputs.iter()) {
                match_node(graph, child_id, child_pat, false, consumers, binds, scalars)?;
            }
            Some(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::DType;

    #[test]
    fn op_to_tag_projects_functional_ops_and_skips_structural() {
        // Functional ops project; the GeluErf/Gelu distinction is preserved.
        assert_eq!(op_to_tag(&Op::Add), Some(OpTag::Add));
        assert_eq!(op_to_tag(&Op::GeluErf), Some(OpTag::GeluErf));
        assert_eq!(op_to_tag(&Op::Gelu), Some(OpTag::Gelu)); // tanh-approx, distinct
        assert_ne!(op_to_tag(&Op::Gelu), op_to_tag(&Op::GeluErf));
        assert_eq!(op_to_tag(&Op::AddScalar(1.0)), Some(OpTag::AddScalar));
        assert_eq!(op_to_tag(&Op::MatMul), Some(OpTag::MatMul));
        // In-place + structural ops are not region nodes.
        assert_eq!(op_to_tag(&Op::ReluInplace), None);
        assert_eq!(op_to_tag(&Op::Const), None);
        assert_eq!(op_to_tag(&Op::Release), None);
    }

    // ---- the structural matcher (match_region) -------------------------------

    fn consumer_counts(g: &Graph) -> std::collections::HashMap<NodeId, usize> {
        let mut c = std::collections::HashMap::new();
        for i in 0..g.len() {
            for &inp in &g.node(NodeId(i)).inputs {
                *c.entry(inp).or_insert(0) += 1;
            }
        }
        c
    }

    fn leaf(g: &mut Graph, s: &fuel_ir::Shape) -> NodeId {
        g.push(crate::Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 })
    }
    fn op1(g: &mut Graph, op: Op, x: NodeId, s: &fuel_ir::Shape) -> NodeId {
        g.push(crate::Node { op, inputs: vec![x], shape: s.clone(), dtype: DType::F32 })
    }
    fn op2(g: &mut Graph, op: Op, x: NodeId, y: NodeId, s: &fuel_ir::Shape) -> NodeId {
        g.push(crate::Node { op, inputs: vec![x, y], shape: s.clone(), dtype: DType::F32 })
    }

    fn relu_add_pattern() -> PatternNode {
        PatternNode::Op {
            op: OpTag::Relu,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::Add,
                attrs: OpAttrs::default(),
                operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
            }],
        }
    }

    #[test]
    fn match_region_binds_relu_add() {
        let mut g = Graph::new();
        let s = fuel_ir::Shape::from_dims(&[2]);
        let a = leaf(&mut g, &s);
        let b = leaf(&mut g, &s);
        let sum = op2(&mut g, Op::Add, a, b, &s);
        let r = op1(&mut g, Op::Relu, sum, &s);
        let counts = consumer_counts(&g);
        let got = match_region(&g, r, &relu_add_pattern(), &|n| *counts.get(&n).unwrap_or(&0));
        assert_eq!(got, Some(vec![a, b]));
    }

    #[test]
    fn match_region_commutative_order_independent() {
        // mul(c, relu(d)); the pattern puts relu on operand[0] — matches via
        // commutativity (§3a.2a).
        let mut g = Graph::new();
        let s = fuel_ir::Shape::from_dims(&[2]);
        let c = leaf(&mut g, &s);
        let d = leaf(&mut g, &s);
        let rd = op1(&mut g, Op::Relu, d, &s);
        let prod = op2(&mut g, Op::Mul, c, rd, &s);
        let pat = PatternNode::Op {
            op: OpTag::Mul,
            attrs: OpAttrs::default(),
            operands: vec![
                PatternNode::Op {
                    op: OpTag::Relu,
                    attrs: OpAttrs::default(),
                    operands: vec![PatternNode::Bind { index: 0 }],
                },
                PatternNode::Bind { index: 1 },
            ],
        };
        let counts = consumer_counts(&g);
        let got = match_region(&g, prod, &pat, &|n| *counts.get(&n).unwrap_or(&0));
        assert_eq!(got, Some(vec![d, c]), "bind 0 = d (under relu), bind 1 = c");
    }

    #[test]
    fn match_region_rejects_wrong_interior_op() {
        // relu(mul(a, b)) does not match the relu(add(...)) pattern.
        let mut g = Graph::new();
        let s = fuel_ir::Shape::from_dims(&[2]);
        let a = leaf(&mut g, &s);
        let b = leaf(&mut g, &s);
        let prod = op2(&mut g, Op::Mul, a, b, &s);
        let r = op1(&mut g, Op::Relu, prod, &s);
        let counts = consumer_counts(&g);
        assert_eq!(
            match_region(&g, r, &relu_add_pattern(), &|n| *counts.get(&n).unwrap_or(&0)),
            None
        );
    }

    #[test]
    fn match_region_declines_shared_interior() {
        // sum feeds two consumers → fusing duplicates it → decline (§3a.4).
        let mut g = Graph::new();
        let s = fuel_ir::Shape::from_dims(&[2]);
        let a = leaf(&mut g, &s);
        let b = leaf(&mut g, &s);
        let sum = op2(&mut g, Op::Add, a, b, &s);
        let r1 = op1(&mut g, Op::Relu, sum, &s);
        let _r2 = op1(&mut g, Op::Neg, sum, &s); // second consumer of `sum`
        let counts = consumer_counts(&g);
        assert_eq!(
            match_region(&g, r1, &relu_add_pattern(), &|n| *counts.get(&n).unwrap_or(&0)),
            None
        );
    }

    // ---- attr-comparison (F1: match_node compares OpAttrs) --------------------

    /// A single-`Permute` region binding one input, with the given absolute
    /// perm on `attrs.perm`. An empty `perm` (`&[]`) is the attr-agnostic
    /// (wildcard) pattern — the shape every existing authored pattern has.
    fn permute_pattern(perm: &[u8]) -> PatternNode {
        PatternNode::Op {
            op: OpTag::Permute,
            attrs: OpAttrs { perm: perm.to_vec(), ..OpAttrs::default() },
            operands: vec![PatternNode::Bind { index: 0 }],
        }
    }

    #[test]
    fn match_node_discriminates_on_perm_attr() {
        // A graph node that permutes with perm = [1, 0].
        let mut g = Graph::new();
        let s = fuel_ir::Shape::from_dims(&[2, 3]);
        let x = leaf(&mut g, &s);
        let p = op1(&mut g, Op::Permute(vec![1, 0]), x, &fuel_ir::Shape::from_dims(&[3, 2]));
        let counts = consumer_counts(&g);
        let cf = |n: NodeId| *counts.get(&n).unwrap_or(&0);

        // The matching perm binds; the non-matching perm is rejected.
        assert_eq!(
            match_region(&g, p, &permute_pattern(&[1, 0]), &cf),
            Some(vec![x]),
            "perm=[1,0] pattern must match a [1,0] graph node",
        );
        assert_eq!(
            match_region(&g, p, &permute_pattern(&[0, 2, 1]), &cf),
            None,
            "perm=[0,2,1] pattern must NOT match a [1,0] graph node (attr discrimination)",
        );

        // No-regression guard: an empty-attr (wildcard) pattern — the shape of
        // every existing authored pattern — still matches regardless of the
        // graph node's real perm. This is what keeps attr-agnostic patterns
        // matching after attrs become comparable.
        assert_eq!(
            match_region(&g, p, &permute_pattern(&[]), &cf),
            Some(vec![x]),
            "empty-perm (wildcard) pattern must still match (no regression)",
        );
    }

    // ---- scalar extraction (the `extract:` layer, §5.3) -----------------------

    #[test]
    fn match_region_extract_reads_live_scalars_in_pattern_pre_order() {
        // Graph: add_scalar(mul_scalar(x, 2.5), 0.5) — two live scalars.
        let mut g = Graph::new();
        let s = fuel_ir::Shape::from_dims(&[2]);
        let x = leaf(&mut g, &s);
        let ms = op1(&mut g, Op::MulScalar(2.5), x, &s);
        let asn = op1(&mut g, Op::AddScalar(0.5), ms, &s);
        let counts = consumer_counts(&g);
        let cf = |n: NodeId| *counts.get(&n).unwrap_or(&0);

        // Slot template: both scalar attrs left empty (open slots).
        let pat = PatternNode::Op {
            op: OpTag::AddScalar,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::MulScalar,
                attrs: OpAttrs::default(),
                operands: vec![PatternNode::Bind { index: 0 }],
            }],
        };
        let (binds, scalars) = match_region_extract(&g, asn, &pat, &cf).expect("matches");
        assert_eq!(binds, vec![x]);
        // Pattern PRE-order: the AddScalar (root) slot before the MulScalar slot.
        assert_eq!(scalars, vec![0.5, 2.5], "live values in pattern pre-order");
    }

    #[test]
    fn baked_scalar_is_a_pattern_constant_not_a_slot() {
        let mut g = Graph::new();
        let s = fuel_ir::Shape::from_dims(&[2]);
        let x = leaf(&mut g, &s);
        let ms = op1(&mut g, Op::MulScalar(2.5), x, &s);
        let counts = consumer_counts(&g);
        let cf = |n: NodeId| *counts.get(&n).unwrap_or(&0);

        let baked = |v: f64| PatternNode::Op {
            op: OpTag::MulScalar,
            attrs: OpAttrs { scalars: vec![v], ..OpAttrs::default() },
            operands: vec![PatternNode::Bind { index: 0 }],
        };
        // The equal baked value matches and extracts NOTHING…
        let (binds, scalars) = match_region_extract(&g, ms, &baked(2.5), &cf).expect("matches");
        assert_eq!(binds, vec![x]);
        assert!(scalars.is_empty(), "baked value is a constant of the pattern, not a slot");
        // …and a different baked value refuses to match at all (attr guard).
        assert!(match_region_extract(&g, ms, &baked(3.0), &cf).is_none());
    }
}
