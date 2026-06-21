//! Kernel-seam JIT-on-request foundation types (kernel-seam-interop §5;
//! fkc-fusion-patterns §3). The frozen wire types both halves reconcile
//! against: the operand projection [`OperandDesc`], the functional-op
//! vocabulary [`OpTag`], and the declarative subgraph grammar [`PatternNode`].
//!
//! These are the §D foundations Baracuda's synthesizer is waiting on: one
//! [`PatternNode`] serves the **JIT region** (Fuel → Baracuda, "build a kernel
//! for this subgraph"), `pattern:` **matching** (a contract's re-fuse rule),
//! and a synthesized op's **`decompose`** (the region re-emitted). The byte
//! form is Fuel's; the synthesizer maps its native records onto these.

use crate::Op;
use fuel_core_types::DType;

// ===========================================================================
// OperandDesc — the raw structure_key projection (Baracuda reconcile §1)
// ===========================================================================

/// The minimal **raw** per-operand projection the schedule key (`structure_key`)
/// is derived from — strides + alignment + extents + dtype, carried verbatim.
/// **Fuel never classifies it**: the contiguity / vector-width / inner-extent /
/// flipped / index-width flags are the *output* of the synthesizer's
/// `structure_key`, not an input here (the ratified single-classifier division,
/// kernel-seam-interop §4.4). This is the inputs-then-output element of a
/// [`crate::jit`] request's operand list.
#[derive(Clone, Debug, PartialEq)]
pub struct OperandDesc {
    /// Logical rank (number of valid entries in `shape`/`strides`).
    pub rank: u8,
    /// Logical extents; symbolic axes carry their capacity bound. Up to 8 dims.
    pub shape: [i64; 8],
    /// Signed element strides (`0` = broadcast, `< 0` = flipped/reversed).
    pub strides: [i64; 8],
    /// Element dtype (the FDX §5 base vocabulary).
    pub dtype: DType,
    /// Base-pointer alignment in bytes — drives the kernel's vector width.
    pub align_bytes: u32,
    /// Quantization facts, carried but not keyed-on in Profile v1.
    pub quant: Option<QuantFacts>,
    /// Symbolic-extent facts (live-vs-capacity / attention-class), carried.
    pub symbolic: Option<SymExtent>,
}

/// Quantization facts carried on an [`OperandDesc`] (Profile v1 carries, does
/// not key on it). Aligns with the FDX `FDXQuant` sidecar projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantFacts {
    /// FDX quant-family code (GGML block / MX / affine-int / affine-float / …).
    pub family: u8,
    /// Block size for block-scaled families (0 if not block-scaled).
    pub block_size: u32,
}

/// A symbolic axis's facts (Phase-D live-vs-capacity), carried on an
/// [`OperandDesc`]. The `shape` entry holds the capacity; this names the axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SymExtent {
    /// The symbolic axis id (`SymId`) this extent is bound to.
    pub sym_id: u32,
    /// The capacity bound (the static maximum the buffer is allocated for).
    pub capacity: i64,
}

// ===========================================================================
// OpTag — the frozen functional-Op vocabulary (kernel-seam-interop §4.1)
// ===========================================================================

/// The §4.1 graph-`Op` vocabulary, **functional ops only** — the stable op
/// identifier a [`PatternNode`] carries. Excludes in-place variants (a region
/// is the *functional* subgraph; in-place is a Fuel-side scheduling rewrite)
/// and structural / bookkeeping ops (`Const`, `Release`, `Alloc`, views, …).
/// [`OpTag::from_op`] is the `Op → OpTag` projection; the inverse direction
/// (which params an emitted op carries) rides [`OpAttrs`] + the `extract:` path.
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
    // comparison (→ U8 mask)
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

impl OpTag {
    /// Project a graph [`Op`] to its functional [`OpTag`]. Returns `None` for
    /// in-place variants, structural/bookkeeping ops, and `Op::Fused` (a fused
    /// op isn't a region node — its *decomposition* is). A `None` op in a
    /// region is an honest "outside the vocabulary" miss, never a crash.
    pub fn from_op(op: &Op) -> Option<OpTag> {
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
            // In-place variants, structural / bookkeeping ops, and Op::Fused
            // are not region nodes.
            _ => return None,
        })
    }
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
/// **region** (Fuel → Baracuda) populates `Op { op, operands, attrs }` +
/// `Bind`; an emitted **`pattern:`** (Baracuda → Fuel) additionally carries the
/// consumer/`extract` routing the matcher compiler reads (landed alongside the
/// matcher). `SeeThrough`/`Any` are matcher-only and never appear in a concrete
/// region.
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
    fn optag_projects_functional_ops_and_skips_structural() {
        // Functional ops project; the GeluErf/Gelu distinction is preserved.
        assert_eq!(OpTag::from_op(&Op::Add), Some(OpTag::Add));
        assert_eq!(OpTag::from_op(&Op::GeluErf), Some(OpTag::GeluErf));
        assert_eq!(OpTag::from_op(&Op::Gelu), Some(OpTag::Gelu)); // tanh-approx, distinct
        assert_ne!(OpTag::from_op(&Op::Gelu), OpTag::from_op(&Op::GeluErf));
        assert_eq!(OpTag::from_op(&Op::AddScalar(1.0)), Some(OpTag::AddScalar));
        assert_eq!(OpTag::from_op(&Op::MatMul), Some(OpTag::MatMul));
        // In-place + structural ops are not region nodes.
        assert_eq!(OpTag::from_op(&Op::ReluInplace), None);
        assert_eq!(OpTag::from_op(&Op::Const), None);
        assert_eq!(OpTag::from_op(&Op::Release), None);
    }

    #[test]
    fn pattern_node_region_for_relu_a_plus_b() {
        // The increment-1 example: relu(a + b) — a 1-output region over 2 inputs.
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
        // Bind indices form [0, 2) — exactly the region's 2 external inputs.
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
