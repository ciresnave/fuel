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
//!
//! The handshake envelope (`SeamHello`, §3) lives in the sibling
//! `fuel-kernel-seam-announce` crate, not here — a provider that only speaks
//! capability negotiation shouldn't need to pull in this region grammar.

/// The §6.20 shape-expression AST + canonical wire codec + evaluator — the
/// KISS-consistent shape vocabulary, homed here (std-only) so `fuel-graph` can
/// carry `Dim`/`ShapeExpr` in [`OpAttrs`] without depending on `fuel-dispatch`
/// (which re-exports this module at `fkc::shape_expr` for its FKC importer).
pub mod shape_expr;

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
    /// Permute/Transpose: the new axis order, **ABSOLUTE** — a permutation of
    /// `0..rank` with `out.axis[i] = in.axis[perm[i]]` (matches Fuel
    /// `Op::Permute` semantics exactly; `Op::Transpose` is the last-two-axes
    /// special case). Empty ⇒ not a permuting node (a matcher wildcard). See
    /// `baracuda-layout-fusion-response.md` F1/F2a.
    pub perm: Vec<u8>,
    /// BroadcastTo/Reshape: the target **LOGICAL** shape (the op's output
    /// shape); one field serves both (the [`OpTag`] disambiguates which op).
    /// Empty ⇒ not a shape-target node (a matcher wildcard).
    pub target_shape: Vec<i64>,
    /// Squeeze/Unsqueeze: the affected dim list (0-based, output-rank terms).
    /// Fuel's single-dim `Op::Squeeze`/`Op::Unsqueeze` emit a one-element list.
    /// Empty ⇒ not a squeeze/unsqueeze node (a matcher wildcard).
    pub dims: Vec<u8>,
    /// `Cast` target dtype as the stable `DType::as_str()` name (dep-free:
    /// this crate can't reference `fuel_ir::DType`; fuel-graph maps back via
    /// `DType::from_str`). Also carries `MaskedFill`'s value dtype. `None` ⇒
    /// not a dtype-carrying node (a matcher wildcard). Convergence Increment A.
    pub cast_dtype: Option<String>,
    /// `Slice` start offset along `axis` (the dim rides the existing `axis`
    /// field). `None` ⇒ not a slice node. Convergence Increment A.
    pub slice_start: Option<u64>,
    /// `Slice` length along `axis`. `None` ⇒ not a slice node. Convergence
    /// Increment A.
    pub slice_len: Option<u64>,
    /// `Roll` cyclic shift along `axis` (signed; the dim rides `axis`). `None`
    /// ⇒ not a roll node. Convergence Increment A.
    pub roll_shift: Option<i64>,
    /// `Pad` per-axis `(before, after)` amounts, one pair per input dim. Empty
    /// ⇒ not a pad node. Convergence Increment A.
    pub pad_amounts: Vec<(u64, u64)>,
    /// `Pad` fill mode code: `0=Constant, 1=Reflect, 2=Replicate` (mirrors
    /// Fuel's `PadMode` order; dep-free integer code). `None` ⇒ not a pad node.
    /// Convergence Increment A.
    pub pad_mode: Option<u8>,
    /// `Pad` constant fill value (used by `PadMode::Constant`). `None` ⇒ not a
    /// pad node. Convergence Increment A.
    pub pad_value: Option<f64>,
    /// Reduction `keepdim` flag (§6.19 reduce-schema conformance — serialized
    /// in the canonical blob). NOT consumed by `tag_to_op`: Fuel's reduce Ops
    /// encode keepdim structurally (`ReduceSumTo`/`ReduceMaxTo` carry the kept
    /// shape; `SumDim`/`MeanDim` are rank-reducing). Convergence Increment A.
    pub keepdim: Option<bool>,
}

// --- §6.19-shaped canonical positional-blob serialization (Convergence Increment A) ---
//
// Little-endian byte writers. `OpAttrs::to_canonical_bytes` emits a per-op
// **positional** body (no field names, no elision — the OpTag fixes the schema)
// then length-prefixes it with a `u32` LE byte length, so an empty-schema op has
// exactly one canonical form (`[0,0,0,0]`). std-only (no `fuel_ir`).
//
// SCOPE (do not overclaim): this is the §6.19 positional *shape*, and it is
// byte-comparable with a Baracuda-emitted blob **for the positionally-conformant
// ops** — elementwise, cast, slice, concat, roll, pad, flip, iota, permute,
// (un)squeeze, shape-target. Two known divergences from the confirmed §6.19.3
// schemas (see docs/outreach/baracuda-recipe-grammar-codesign-reply-2.md), which
// the pinned node schema `Op{op_name, op_attrs, child_edges}` reconciles WITHOUT
// widening this blob:
//   * `reduce{monoid, reduce_axes, keepdim}` — Fuel emits single-axis
//     `{axis, keepdim}`. `monoid` rides `op_name` (distinct SumDim/MaxDim/MinDim
//     tags), and a multi-axis `reduce_axes` LIST is DEFERRED (Fuel models
//     single-axis reduce; no consumer yet).
//   * `gather/scatter{axis, oob_policy, index_operand, index_dtype, scatter_combine}`
//     — Fuel emits `{axis}`. `scatter_combine` rides `op_name` (IndexAdd vs
//     ScatterAdd), `index_operand` rides `child_edges`, `index_dtype` rides that
//     operand node; `oob_policy` is a DEFERRED unwired slot (no carrier yet).
// See kernel-seam-interop.md §7.3.2 for the per-op field-order table + this scope.

fn put_u32(b: &mut Vec<u8>, x: u32) { b.extend_from_slice(&x.to_le_bytes()); }
fn put_u64(b: &mut Vec<u8>, x: u64) { b.extend_from_slice(&x.to_le_bytes()); }
fn put_i64(b: &mut Vec<u8>, x: i64) { b.extend_from_slice(&x.to_le_bytes()); }
fn put_f64(b: &mut Vec<u8>, x: f64) { b.extend_from_slice(&x.to_le_bytes()); }
fn put_str(b: &mut Vec<u8>, s: &str) { put_u32(b, s.len() as u32); b.extend_from_slice(s.as_bytes()); }
fn put_i64_list(b: &mut Vec<u8>, xs: &[i64]) { put_u32(b, xs.len() as u32); for &x in xs { put_i64(b, x); } }
fn put_u32_list(b: &mut Vec<u8>, xs: &[u32]) { put_u32(b, xs.len() as u32); for &x in xs { put_u32(b, x); } }
fn put_f64_list(b: &mut Vec<u8>, xs: &[f64]) { put_u32(b, xs.len() as u32); for &x in xs { put_f64(b, x); } }

impl OpAttrs {
    /// Serialize these attrs to the KISS §6.19 canonical positional blob for
    /// `op`: a per-op **positional** little-endian body (no elision — the
    /// `OpTag` determines the fixed schema), length-prefixed with a `u32` LE
    /// byte count. An op whose schema is empty (`Add`, `Neg`, `MatMul`, `Where`,
    /// comparisons, …) serializes as the single canonical form `[0,0,0,0]`.
    /// Deterministic + dependency-free.
    ///
    /// **Conformance scope (do not overclaim):** byte-comparable with a
    /// Baracuda-emitted blob for the positionally-conformant ops (elementwise,
    /// cast, slice, concat, roll, pad, flip, iota, permute, (un)squeeze,
    /// shape-target). `reduce` emits Fuel's single-axis `{axis, keepdim}` and
    /// `gather`/`scatter` emit `{axis}`; `oob_policy` and a multi-axis
    /// `reduce_axes` list are DEFERRED (no carrier/consumer yet), while
    /// `monoid`/`scatter_combine` ride `op_name` and the index operand/dtype
    /// ride `child_edges`/that operand node per the pinned node schema — so they
    /// legitimately do not belong in this blob. See the module comment above and
    /// kernel-seam-interop.md §7.3.2.
    ///
    /// M-3: the `unwrap_or(...)` defaults below cannot distinguish an *unset*
    /// field from a genuine zero (e.g. `axis: None` vs `Some(0)`). Harmless
    /// today — there is no decoder; this is a forward-serialization only, and an
    /// op that reaches a given arm always has the field set (`op_to_attrs` /
    /// `tag_to_op` guarantee it). A future decoder must not round-trip `None`.
    pub fn to_canonical_bytes(&self, op: OpTag) -> Vec<u8> {
        use OpTag as T;
        let mut body: Vec<u8> = Vec::new();
        match op {
            // Shape-target ops: the logical output shape (Iota's len rides it).
            T::Reshape | T::BroadcastTo | T::ReduceSumTo | T::ReduceMaxTo | T::Iota => {
                put_i64_list(&mut body, &self.target_shape);
            }
            // Permute/Transpose: the absolute axis order.
            T::Permute | T::Transpose => {
                let perm: Vec<u32> = self.perm.iter().map(|&p| p as u32).collect();
                put_u32_list(&mut body, &perm);
            }
            // Squeeze/Unsqueeze: the affected dim list.
            T::Unsqueeze | T::Squeeze => {
                let dims: Vec<u32> = self.dims.iter().map(|&d| d as u32).collect();
                put_u32_list(&mut body, &dims);
            }
            // Slice: axis(u32), start(u64), len(u64).
            T::Slice => {
                put_u32(&mut body, self.axis.unwrap_or(0) as u32);
                put_u64(&mut body, self.slice_start.unwrap_or(0));
                put_u64(&mut body, self.slice_len.unwrap_or(0));
            }
            // Single-axis ops (dim rides `axis`).
            T::Concat | T::Flip | T::Triu | T::Tril
            | T::IndexSelect | T::Gather | T::IndexAdd | T::ScatterAdd => {
                put_i64(&mut body, self.axis.unwrap_or(0));
            }
            // Roll: axis(i64) + shift(i64).
            T::Roll => {
                put_i64(&mut body, self.axis.unwrap_or(0));
                put_i64(&mut body, self.roll_shift.unwrap_or(0));
            }
            // Dim reductions + cumsum: axis(i64) + keepdim(u8).
            T::SumDim | T::MeanDim | T::CumSum => {
                put_i64(&mut body, self.axis.unwrap_or(0));
                body.push(self.keepdim.unwrap_or(false) as u8);
            }
            // Cast: length-prefixed dtype name.
            T::Cast => put_str(&mut body, self.cast_dtype.as_deref().unwrap_or("")),
            // Pad: amounts (count + (before:u64, after:u64) each) + mode(u8) + value(f64).
            T::Pad => {
                put_u32(&mut body, self.pad_amounts.len() as u32);
                for &(before, after) in &self.pad_amounts {
                    put_u64(&mut body, before);
                    put_u64(&mut body, after);
                }
                body.push(self.pad_mode.unwrap_or(0));
                put_f64(&mut body, self.pad_value.unwrap_or(0.0));
            }
            // Scalar-param ops: the scalar list.
            T::AddScalar | T::MulScalar | T::Clamp | T::PowI => {
                put_f64_list(&mut body, &self.scalars);
            }
            // MaskedFill: scalar value(s) + value dtype name.
            T::MaskedFill => {
                put_f64_list(&mut body, &self.scalars);
                put_str(&mut body, self.cast_dtype.as_deref().unwrap_or(""));
            }
            // Empty-schema ops (elementwise, comparison, Where, MatMul, scalar
            // reductions, log-softmax, …) and any tag added later: zero-length.
            _ => {}
        }
        let mut out = (body.len() as u32).to_le_bytes().to_vec();
        out.extend_from_slice(&body);
        out
    }
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

    // ---- Task 7: §6.19 canonical positional-blob serialization --------------

    #[test]
    fn empty_schema_op_serializes_zero_length() {
        // Add carries no attrs → one canonical byte form: u32 LE length 0.
        assert_eq!(OpAttrs::default().to_canonical_bytes(OpTag::Add), vec![0, 0, 0, 0]);
        assert_eq!(OpAttrs::default().to_canonical_bytes(OpTag::MatMul), vec![0, 0, 0, 0]);
    }

    #[test]
    fn slice_serializes_positionally() {
        // Slice schema (positional): axis(u32), start(u64), len(u64) — see kernel-seam-interop.md.
        let a = OpAttrs { axis: Some(1), slice_start: Some(2), slice_len: Some(3), ..OpAttrs::default() };
        let mut expect = Vec::new();
        let body = {
            let mut b = Vec::new();
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&2u64.to_le_bytes());
            b.extend_from_slice(&3u64.to_le_bytes());
            b
        };
        expect.extend_from_slice(&(body.len() as u32).to_le_bytes());
        expect.extend_from_slice(&body);
        assert_eq!(a.to_canonical_bytes(OpTag::Slice), expect);
    }

    #[test]
    fn cast_serializes_dtype_name_length_prefixed() {
        let a = OpAttrs { cast_dtype: Some("f16".into()), ..OpAttrs::default() };
        let mut body = Vec::new();
        body.extend_from_slice(&(3u32.to_le_bytes())); // name length
        body.extend_from_slice(b"f16");
        let mut expect = (body.len() as u32).to_le_bytes().to_vec();
        expect.extend_from_slice(&body);
        assert_eq!(a.to_canonical_bytes(OpTag::Cast), expect);
    }

    #[test]
    fn canonical_bytes_are_deterministic() {
        let a = OpAttrs { target_shape: vec![2, 3], ..OpAttrs::default() };
        assert_eq!(a.to_canonical_bytes(OpTag::Reshape), a.to_canonical_bytes(OpTag::Reshape));
    }

    // ---- Per-carrier width conformance (KISS coordinator pin, vs KISS main c9153b2) --
    //
    // There are THREE coexisting op_attrs / shape-expr framings. Widths are pinned
    // PER-CARRIER, never as "the op_attrs width" — a future consolidation must NOT
    // silently unify them (KISS #67 do-not-unify):
    //   (a) #67 NODE-ENVELOPE op_attrs        → u32-LE OUTER byte length, payload
    //       verbatim, no-parse-inside (§6.19-0010). Live producer:
    //       `OpAttrs::to_canonical_bytes`.
    //   (b) KISS-Grammar §6.8-0007 REGION-NODE-TABLE op_attrs SUB-BLOCK → u16-LE
    //       length + payload verbatim; EMPTY = 0x0000. A DIFFERENT carrier from (a).
    //       Fuel ships NO producer yet (node/table wire serializer is #67-gated,
    //       slice 4); this pin binds that future serializer to u16-LE here.
    //   (c) §6.20-0005 SHAPE-EXPR binary-node CHILD length → u16-LE. Live producer:
    //       `shape_expr::Dim::encode`.
    #[test]
    fn three_carrier_width_pins_stay_distinct() {
        use crate::shape_expr::{Dim, LAST};

        // Carrier (a): node-envelope op_attrs — outer prefix is EXACTLY 4 bytes
        // (u32-LE); an empty schema is the 4-byte zero form.
        let empty = OpAttrs::default().to_canonical_bytes(OpTag::Add);
        assert_eq!(empty, vec![0u8, 0, 0, 0], "carrier (a): empty node-envelope op_attrs = u32-LE zero (4 bytes)");
        let sliced = OpAttrs { axis: Some(1), slice_start: Some(2), slice_len: Some(3), ..OpAttrs::default() };
        let blob = sliced.to_canonical_bytes(OpTag::Slice);
        let a_body_len = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize;
        assert_eq!(blob.len(), 4 + a_body_len, "carrier (a): outer length prefix is u32-LE (4 bytes), body verbatim");

        // Carrier (b): region-node-table op_attrs sub-block — u16-LE length +
        // verbatim payload; empty = 0x0000. Modeled here (no Fuel producer yet) so
        // the width pin is executable and NAMED before the slice-4 serializer lands.
        fn region_node_table_op_attrs_sub_block(payload: &[u8]) -> Vec<u8> {
            let mut out = (payload.len() as u16).to_le_bytes().to_vec();
            out.extend_from_slice(payload); // verbatim, no-parse-inside
            out
        }
        assert_eq!(
            region_node_table_op_attrs_sub_block(&[]),
            vec![0x00, 0x00],
            "carrier (b): EMPTY region-node-table op_attrs sub-block = 0x0000 (u16-LE, 2 bytes)"
        );
        assert_eq!(
            region_node_table_op_attrs_sub_block(&[0xAB, 0xCD]),
            vec![0x02, 0x00, 0xAB, 0xCD],
            "carrier (b): sub-block length prefix is u16-LE (2 bytes), payload verbatim"
        );

        // Carrier (c): shape-expr binary-node child length — u16-LE (2 bytes),
        // inside Dim::{Add,Sub,Mul,Div}. The rope-half golden's child prefixes:
        // tag(0x08) ++ u16-LE(3) ++ Extent(3B) ++ u16-LE(9) ++ Const(9B).
        let half = Dim::Div(
            Box::new(Dim::Extent { operand: 0, axis: LAST }),
            Box::new(Dim::Const(2)),
        );
        let bytes = half.encode();
        assert_eq!(
            [bytes[1], bytes[2]],
            3u16.to_le_bytes(),
            "carrier (c): first child length prefix is u16-LE (2 bytes)"
        );
        assert_eq!(
            [bytes[6], bytes[7]],
            9u16.to_le_bytes(),
            "carrier (c): second child length prefix is u16-LE (2 bytes)"
        );
        assert_eq!(bytes.len(), 1 + 2 + 3 + 2 + 9, "carrier (c): whole rope-half blob accounted for");

        // Pinned widths, side by side — (a)=4 (u32-LE) vs (b)=2 (u16-LE) vs
        // (c)=2 (u16-LE). (b) and (c) sharing a width is coincidence, not unity:
        // each is pinned by its OWN carrier name above.
    }
}
