//! Frozen kernel-seam wire types — the JIT region / declarative-pattern grammar
//! shared across the Fuel <-> backend-synthesizer seam (kernel-seam-interop
//! §3/§5; fkc-fusion-patterns §3). **Types only, no logic, no Fuel-graph
//! dependency**: a synthesizer backend (e.g. Baracuda) depends on this small
//! crate; the `Op -> OpTag` projection (`fuel_graph::jit::op_to_tag`) and the
//! structural matcher (`fuel_graph::jit::match_region`) stay Fuel-side because
//! they need the graph.
//!
//! One [`PatternNode`] serves three roles: the **JIT region** (Fuel ->
//! synthesizer, "build a kernel for this subgraph"), a contract's `pattern:`
//! **re-fuse rule**, and a synthesized op's **`decompose`** (the region
//! re-emitted). The operand-side projection (`OperandDesc`) is the synthesizer's
//! `structure_key` input and lives in its types crate — not here.

// ===========================================================================
// OpTag — the frozen functional-Op vocabulary (kernel-seam-interop §4.1)
// ===========================================================================

/// The §4.1 graph-`Op` vocabulary, **functional ops only** — the stable op
/// identifier a [`PatternNode`] carries. Excludes in-place variants (a region
/// is the *functional* subgraph; in-place is a Fuel-side scheduling rewrite)
/// and structural / bookkeeping ops (`Const`, `Release`, `Alloc`, views, ...).
/// The `Op -> OpTag` projection (`op_to_tag`) lives Fuel-side (it needs the
/// graph `Op`); the inverse — which params an emitted op carries — rides
/// [`OpAttrs`] + the `extract:` path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OpTag {
    // binary arithmetic / extremum
    Add, Sub, Mul, Div, Maximum, Minimum, Pow, Rem,
    // unary math
    Neg, Abs, Sqr, Sqrt, Rsqrt, Recip, Exp, Log, Sin, Cos,
    // activations (Gelu = tanh-approx; GeluErf = exact erf — distinct, §3 note)
    Tanh, Sigmoid, Silu, Gelu, GeluErf, Relu, Erf, Step,
    // rounding / sign
    Floor, Ceil, Round, Sign,
    // scalar-param (value param-ized; attrs carries the slot)
    AddScalar, MulScalar, PowI, Clamp,
    // comparison (-> U8 mask)
    Equal, Ne, Lt, Le, Gt, Ge,
    // select / mask
    Where, MaskedFill,
    // reductions
    SumAll, MaxAll, MinAll, MeanAll, SumDim, MeanDim, ReduceSumTo, ReduceMaxTo, CumSum,
    // matmul
    MatMul,
    // shape / layout (metadata or copy)
    Transpose, Permute, Reshape, BroadcastTo, Unsqueeze, Squeeze, Cast, Slice, Concat, Flip, Roll, Pad, Triu, Tril,
    // indexing / gather-scatter
    IndexSelect, Gather, IndexAdd, ScatterAdd,
    // fused-primitive helpers
    LogSoftmaxLastDim,
    // value source
    Iota,
}

// ===========================================================================
// PatternNode — the §3 declarative subgraph grammar
// ===========================================================================

/// Non-tensor attributes a [`PatternNode::Op`] carries (fkc-fusion-patterns
/// §3a.4; Baracuda reconcile §2). For scalar-param ops the value is **not
/// baked** — it identifies the slot the emitted `extract:` path points at, and
/// the matcher re-reads the live value from the matched graph node at match
/// time. Carries the load-bearing attributes the general vocabulary needs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OpAttrs {
    /// Scalar value(s) for scalar-param ops (`AddScalar`/`MulScalar`/`Clamp`),
    /// as the region's snapshot of the slot (re-read live via `extract:`).
    pub scalars: Vec<f64>,
    /// Axis attribute for dim-bearing ops (reductions, `Triu`/`Tril` diagonal).
    pub axis: Option<i64>,
}

/// A node of the §3 declarative subgraph grammar. One type, two directions: a
/// **region** (Fuel -> synthesizer) populates `Op { op, operands, attrs }` +
/// `Bind`; an emitted **`pattern:`** (synthesizer -> Fuel) additionally carries
/// the consumer/`extract` routing the matcher compiler reads. `SeeThrough`/`Any`
/// are matcher-only and never appear in a concrete region.
#[derive(Clone, Debug, PartialEq)]
pub enum PatternNode {
    /// An op over the [`OpTag`] vocabulary with one child per tensor input
    /// (ordered, exact arity). `attrs` carries the scalar slot / load-bearing
    /// attributes.
    Op {
        op: OpTag,
        operands: Vec<PatternNode>,
        attrs: OpAttrs,
    },
    /// A leaf: bind the producing node as the fused op's `input[index]`. A
    /// repeated `index` is a node-identity guard on a shared input (§3.2);
    /// indices across a region MUST equal `[0, n_inputs)`.
    Bind { index: u8 },
    /// Match the inner node after skipping zero-or-more transparent wrappers
    /// (§3.3). Matcher-only.
    SeeThrough { then: Box<PatternNode> },
    /// Wildcard — matches any single node (§3.4). Matcher-only.
    Any,
}

impl PatternNode {
    /// Collect the distinct `Bind` indices in this tree (a region's external
    /// inputs). Used to validate `bind` indices form `[0, n_inputs)`.
    pub fn bind_indices(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.collect_binds(&mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    fn collect_binds(&self, out: &mut Vec<u8>) {
        match self {
            PatternNode::Bind { index } => out.push(*index),
            PatternNode::Op { operands, .. } => {
                for o in operands {
                    o.collect_binds(out);
                }
            }
            PatternNode::SeeThrough { then } => then.collect_binds(out),
            PatternNode::Any => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_node_region_for_relu_a_plus_b() {
        // relu(a + b) — a 1-output region over 2 inputs.
        let region = PatternNode::Op {
            op: OpTag::Relu,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::Add,
                attrs: OpAttrs::default(),
                operands: vec![
                    PatternNode::Bind { index: 0 },
                    PatternNode::Bind { index: 1 },
                ],
            }],
        };
        assert_eq!(region.bind_indices(), vec![0, 1]);
    }

    #[test]
    fn shared_input_node_identity_guard() {
        // mul(x, x) — repeated bind: 0 is the shared-input node-identity guard.
        let region = PatternNode::Op {
            op: OpTag::Mul,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 0 }],
        };
        assert_eq!(region.bind_indices(), vec![0]);
    }
}
