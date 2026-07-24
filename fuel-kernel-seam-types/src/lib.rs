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
    // reductions (MaxDim: additive, Increment C slice 1 T4 — the D3 keepdim
    // swap spells keepdim reduces as {Max,Sum,Mean}Dim + Unsqueeze)
    SumAll, MaxAll, MinAll, MeanAll, SumDim, MaxDim, MeanDim, ReduceSumTo, ReduceMaxTo, CumSum,
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

    // --- Shape-RELATIVE recipe-interior fields (Increment C slice 1, D2) ---
    //
    // The four fields below make a recipe polymorphic across shapes/ranks:
    // instead of baking an absolute value they carry an expression over the
    // region's **Bind space** (`ShapeExpr::SameAs { operand: i }` /
    // `Dim::Extent { operand: i, .. }` reference `Bind { index: i }` — the same
    // operand-index convention the merged KISS shape-oracle RFC pins for
    // contracts). They are resolved to the concrete sibling field at emit time
    // (`fuel_graph::runtime_fused::resolve_rel_attrs`); a rel field and its
    // concrete sibling set together is a typed resolution error (mutual
    // exclusion), never a silent precedence.
    //
    // **Deliberately NOT serialized to the §6.19 wire (KISS #67-gated).**
    // `to_canonical_bytes` serializes only the concrete fields: the pinned
    // §6.19 arms for `broadcast_to`/`slice` are ABSOLUTE, and the node-envelope
    // framing that could carry a relative alternative is being defined in
    // KISS #67 — serializing a rel form now would unilaterally extend a shared
    // byte contract (propose-first). Emitted/graph nodes are always concrete
    // post-resolution, so rel-attr recipes never flow to `to_canonical_bytes`
    // callers. Pinned by `rel_attr_fields_are_absent_from_the_6_19_wire`.
    /// Shape-relative alternative to `target_shape`: the target is another
    /// operand's shape (`SameAs { operand }` over the Bind space). `None` ⇒
    /// use `target_shape` (or not a shape-target node).
    pub target_shape_rel: Option<shape_expr::ShapeExpr>,
    /// Shape-relative alternative to `slice_start`: a `DimExpr` over the Bind
    /// space (e.g. rope's `Div(Extent(0, LAST), 2)`). `None` ⇒ use
    /// `slice_start`.
    pub slice_start_rel: Option<shape_expr::Dim>,
    /// Shape-relative alternative to `slice_len`. `None` ⇒ use `slice_len`.
    pub slice_len_rel: Option<shape_expr::Dim>,
    /// Rank-relative alternative to the op's axis carrier (`axis`, or `dims`
    /// for Squeeze/Unsqueeze): this op's axis is its per-tag LAST — resolved
    /// against `rank(operand[0])` as `rank − 1` for reduces/Slice/Concat/Flip/
    /// CumSum/… and Squeeze, and `rank` (append) for Unsqueeze. `false` ⇒ use
    /// the absolute carrier.
    pub axis_last: bool,

    // --- Matmul contraction role vectors (Increment C slice 1, T9/D5) -------
    //
    // The LOCKED reply-3 contraction descriptor (commit `b64aa1db`; §5 of the
    // recipe-grammar design input). `matmul`'s op_attrs is two per-axis role
    // vectors over the roles { Batch=0, FreeM=1, FreeN=2, ContractedK=3 } (one
    // **u8** per axis), `lhs_roles` then `rhs_roles`, each length = operand rank.
    // Role vectors encode WHICH axis plays which role, **not extents** — so GQA
    // (differing-but-divisible batch extents) serializes to identical all-Batch
    // leading roles. Both empty ⇒ the rank-polymorphic implicit form (recipes
    // keep matmul implicit; concrete/ingested nodes get explicit roles). The
    // canonical cell (same-rank ≥ 2; leading Batch; lhs=[..,FreeM,ContractedK],
    // rhs=[..,ContractedK,FreeN]) is derived by [`matmul_roles`] and validated at
    // resolve time (`fuel_graph::runtime_fused::tag_to_op`).
    //
    // Serialized on carrier (a) — `to_canonical_bytes(MatMul)` emits
    // `u32_le(len lhs) ++ lhs_roles ++ u32_le(len rhs) ++ rhs_roles` when set, or
    // the canonical empty body `[00,00,00,00]` when both are empty (preserving
    // today's golden). Baracuda (#68) confirmed the rank-2 golden as the shared
    // cross-producer fixture and has no near-term binary arm, so Fuel's
    // serializer is the contract.
    /// LHS per-axis contraction roles ({Batch=0,FreeM=1,FreeN=2,ContractedK=3},
    /// u8 each; length = lhs rank). Empty ⇒ implicit (rank-polymorphic) matmul.
    pub lhs_roles: Vec<u8>,
    /// RHS per-axis contraction roles (same encoding; length = rhs rank). Empty
    /// ⇒ implicit (rank-polymorphic) matmul.
    pub rhs_roles: Vec<u8>,
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
// (un)squeeze, shape-target, matmul role-vectors (§5, LOCKED). Two known
// divergences from the confirmed §6.19.3
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
fn put_u8_list(b: &mut Vec<u8>, xs: &[u8]) { put_u32(b, xs.len() as u32); b.extend_from_slice(xs); }

/// Derive the canonical matmul role vectors for a same-rank ≥ 2 contraction
/// (the LOCKED reply-3 cell, §5): `lhs = [Batch×(r−2), FreeM, ContractedK]`,
/// `rhs = [Batch×(r−2), ContractedK, FreeN]` over the roles
/// { Batch=0, FreeM=1, FreeN=2, ContractedK=3 }. Roles encode axis POSITIONS,
/// not extents — GQA-divisible batch stays all-`Batch`. Pure + never-panic: a
/// rank < 2 operand (never a real matmul input) yields an all-`Batch` vector of
/// that length, which the resolver's exact-match check rejects as non-canonical.
pub fn matmul_roles(lhs_rank: usize, rhs_rank: usize) -> (Vec<u8>, Vec<u8>) {
    fn one(rank: usize, second_last: u8, last: u8) -> Vec<u8> {
        let mut v = vec![0u8; rank]; // Batch = 0 for every leading axis
        if rank >= 2 {
            v[rank - 2] = second_last;
            v[rank - 1] = last;
        }
        v
    }
    // lhs: [.., FreeM(1), ContractedK(3)]; rhs: [.., ContractedK(3), FreeN(2)].
    (one(lhs_rank, 1, 3), one(rhs_rank, 3, 2))
}

impl OpAttrs {
    /// Serialize these attrs to the KISS §6.19 canonical positional blob for
    /// `op`: a per-op **positional** little-endian body (no elision — the
    /// `OpTag` determines the fixed schema), length-prefixed with a `u32` LE
    /// byte count. An op whose schema is empty (`Add`, `Neg`, `Where`,
    /// comparisons, …) serializes as the single canonical form `[0,0,0,0]`.
    /// `MatMul` is empty-bodied ONLY when its role vectors are unset (the
    /// implicit rank-polymorphic form); explicit roles serialize the LOCKED
    /// §5 contraction descriptor. Deterministic + dependency-free.
    ///
    /// **Conformance scope (do not overclaim):** byte-comparable with a
    /// Baracuda-emitted blob for the positionally-conformant ops (elementwise,
    /// cast, slice, concat, roll, pad, flip, iota, permute, (un)squeeze,
    /// shape-target, matmul role-vectors — the shared cross-producer golden,
    /// Baracuda #68). `reduce` emits Fuel's single-axis `{axis, keepdim}` and
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
            // Dim reductions + cumsum: axis(i64) + keepdim(u8). The monoid
            // rides op_name (distinct SumDim/MaxDim/MeanDim tags), so every
            // reduce tag shares this one row schema.
            T::SumDim | T::MaxDim | T::MeanDim | T::CumSum => {
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
            // MatMul: the LOCKED role-vector contraction descriptor (§5,
            // reply-3) — `u32_le(len lhs) ++ lhs_roles ++ u32_le(len rhs) ++
            // rhs_roles`, u8 roles, lhs-then-rhs. Both empty ⇒ the empty body
            // (the canonical `[00,00,00,00]` implicit form; recipes keep matmul
            // rank-polymorphic). The rank-2 golden is the shared cross-producer
            // fixture (Baracuda #68).
            T::MatMul => {
                if !self.lhs_roles.is_empty() || !self.rhs_roles.is_empty() {
                    put_u8_list(&mut body, &self.lhs_roles);
                    put_u8_list(&mut body, &self.rhs_roles);
                }
            }
            // Empty-schema ops (elementwise, comparison, Where, scalar
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

// ===========================================================================
// Advisory ULP-band reference cases — the single drift-pinning fixture
// ===========================================================================

/// The shared drift-pinning fixture for the **advisory ULP band** formula
/// (KISS-CONFORM §6.8; kiss-ref tolerance refinement 2026-07-23).
///
/// The band formula is hand-maintained in TWO places that cannot be
/// co-compiled on a CPU build:
///
///   * `fuel_kiss_ref_backend::region_advisory_tolerance` (with
///     `op_ulp_ceiling` / `region_ulp_ceilings`) — the **reference** formula,
///     in an adapter crate pulled only under `fuel-dispatch`'s `cuda` feature;
///     and
///   * `fuel_dispatch::jit_ingest::advisory_ulp_band` — the **live** copy on
///     the ingestion path, which must stay testable under `--features jit`
///     alone (i.e. without the cuda-gated adapter).
///
/// Because those two live in crates gated apart, no single CPU test sees both.
/// This table is the shared source of truth each side is pinned to: every
/// `(region, expected)` pair here is asserted from BOTH sides (the adapter's
/// `region.rs` tests and fuel-dispatch's `jit_ingest.rs` tests), so a drift in
/// either formula fails that side's build. It is homed in this dependency-free
/// grammar crate — the one both already depend on — precisely because it is the
/// only place visible to both without touching the cuda gating.
///
/// `expected` is the NORMALIZED band the live consumer applies:
/// `None` ⇒ compare `Tolerance::Exact` (an op-free region *or* a single exact
/// op — both drive an exact comparison), `Some(n)` ⇒ `Tolerance::Ulp(n)`. The
/// adapter's richer `Option<Tolerance>` collapses `None` (op-free) and
/// `Some(Exact)` (single exact op) onto `None` here, matching
/// `advisory_ulp_band`'s native `Option<u64>` shape.
///
/// Baked-in ceilings mirror the two formulas' shared rule: transcendental ops
/// contribute their kiss §6.8 ceiling (Exp/Erf declare 4; kiss non-primitives
/// Tanh/Sigmoid/Silu/Gelu/Rsqrt fall back to 4); IEEE-correctly-rounded
/// Sqrt/Recip are exact-class (no ceiling); an exact-only region of `n` ops
/// bands at `n - 1`; a transcendental region bands at `Σ ceilings +
/// (n_exact - 1)` with the exact term saturating at 0.
pub fn advisory_band_reference_cases() -> Vec<(PatternNode, Option<u64>)> {
    fn bind(i: u8) -> PatternNode {
        PatternNode::Bind { index: i }
    }
    fn op(op: OpTag, operands: Vec<PatternNode>) -> PatternNode {
        PatternNode::Op { op, operands, attrs: OpAttrs::default() }
    }
    vec![
        // Op-free region: nothing to band -> exact comparison.
        (bind(0), None),
        // Single exact op -> exact comparison.
        (op(OpTag::Add, vec![bind(0), bind(1)]), None),
        // Multi-node exact-only region -> Ulp(n_ops - 1).
        (op(OpTag::Relu, vec![op(OpTag::Add, vec![bind(0), bind(1)])]), Some(1)),
        // Deeper exact-only region (3 ops) -> Ulp(3 - 1).
        (
            op(
                OpTag::Relu,
                vec![op(OpTag::Relu, vec![op(OpTag::Add, vec![bind(0), bind(1)])])],
            ),
            Some(2),
        ),
        // Sqrt is IEEE-correctly-rounded -> exact class; sqrt(a+b) = 2 exact ops.
        (op(OpTag::Sqrt, vec![op(OpTag::Add, vec![bind(0), bind(1)])]), Some(1)),
        // Lone transcendental keeps exactly its own ceiling (Exp declares 4).
        (op(OpTag::Exp, vec![bind(0)]), Some(4)),
        // Transcendental + one exact: 4 + (1 - 1) = 4.
        (op(OpTag::Exp, vec![op(OpTag::Add, vec![bind(0), bind(1)])]), Some(4)),
        // Two transcendentals (Exp 4, Tanh fallback 4) + one exact: 8 + 0 = 8.
        (
            op(
                OpTag::Tanh,
                vec![op(OpTag::Add, vec![op(OpTag::Exp, vec![bind(0)]), bind(1)])],
            ),
            Some(8),
        ),
        // All-transcendental exp(tanh(x)): 4 + 4, exact term saturates at 0.
        (op(OpTag::Exp, vec![op(OpTag::Tanh, vec![bind(0)])]), Some(8)),
        // Silu (kiss non-primitive fallback 4) over add(a, exp(b)): 4 + 4 + 0.
        (
            op(
                OpTag::Silu,
                vec![op(OpTag::Add, vec![bind(0), op(OpTag::Exp, vec![bind(1)])])],
            ),
            Some(8),
        ),
    ]
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

    // ---- T2 (Increment C slice 1): shape-relative interior fields stay OFF
    // the §6.19 wire ------------------------------------------------------
    //
    // D2 pin (KISS #67-gated): `target_shape_rel` / `slice_start_rel` /
    // `slice_len_rel` / `axis_last` are IN-MEMORY recipe data, resolved to the
    // concrete fields at emit time. The pinned §6.19 arms for
    // `broadcast_to`/`slice` are ABSOLUTE (`put_i64_list(target_shape)`,
    // `u32(axis) ++ u64(start) ++ u64(len)`), and the node-envelope framing
    // that could carry a relative alternative is being defined in KISS #67 —
    // serializing a rel form now would unilaterally extend a shared byte
    // contract (propose-first says no). This test pins the ABSENCE: the wire
    // bytes are identical with and without every rel field set, for every
    // schema family the rel fields could plausibly leak into.
    #[test]
    fn rel_attr_fields_are_absent_from_the_6_19_wire() {
        use crate::shape_expr::{Dim, LAST, ShapeExpr};
        let half = Dim::Div(
            Box::new(Dim::Extent { operand: 0, axis: LAST }),
            Box::new(Dim::Const(2)),
        );
        let rel_only = OpAttrs {
            target_shape_rel: Some(ShapeExpr::SameAs { operand: 0 }),
            slice_start_rel: Some(half.clone()),
            slice_len_rel: Some(half),
            axis_last: true,
            ..OpAttrs::default()
        };
        // Shape-target arm (BroadcastTo/Reshape): rel-only attrs serialize
        // byte-identically to fully-default attrs (an unset target_shape).
        assert_eq!(
            rel_only.to_canonical_bytes(OpTag::BroadcastTo),
            OpAttrs::default().to_canonical_bytes(OpTag::BroadcastTo),
            "target_shape_rel must not reach the broadcast_to wire arm"
        );
        // Slice arm: the ABSOLUTE fields serialize; adding every rel field on
        // top changes NOTHING.
        let abs = OpAttrs { axis: Some(1), slice_start: Some(2), slice_len: Some(3), ..OpAttrs::default() };
        let abs_plus_rel = OpAttrs { axis: Some(1), slice_start: Some(2), slice_len: Some(3), ..rel_only.clone() };
        assert_eq!(
            abs_plus_rel.to_canonical_bytes(OpTag::Slice),
            abs.to_canonical_bytes(OpTag::Slice),
            "slice_{{start,len}}_rel must not reach the slice wire arm"
        );
        // Reduce row (axis ++ keepdim): axis_last must not leak.
        let sd = OpAttrs { axis: Some(1), ..OpAttrs::default() };
        let sd_plus_rel = OpAttrs { axis: Some(1), ..rel_only.clone() };
        assert_eq!(
            sd_plus_rel.to_canonical_bytes(OpTag::SumDim),
            sd.to_canonical_bytes(OpTag::SumDim),
            "axis_last must not reach the reduce wire row"
        );
        // Empty-schema op: stays the single canonical 4-byte zero form even
        // with every rel field set.
        assert_eq!(rel_only.to_canonical_bytes(OpTag::Add), vec![0, 0, 0, 0]);
    }

    // ---- T4 (Increment C slice 1): additive OpTag::MaxDim -----------------

    #[test]
    fn max_dim_serializes_the_reduce_row() {
        // `OpTag::MaxDim` joins the pinned §6.19 reduce row — carrier (a), the
        // node-envelope op_attrs blob (u32-LE outer): body = i64(axis) ++
        // u8(keepdim). The monoid rides `op_name` (distinct SumDim/MaxDim/
        // MinDim tags — see the module comment's reduce-schema note), so the
        // blob is IDENTICAL in shape to SumDim's. Golden for axis=1, keepdim
        // unset (=0): u32-LE(9) ++ i64-LE(1) ++ 0x00.
        let a = OpAttrs { axis: Some(1), ..OpAttrs::default() };
        let mut expect = 9u32.to_le_bytes().to_vec();
        expect.extend_from_slice(&1i64.to_le_bytes());
        expect.push(0u8);
        assert_eq!(a.to_canonical_bytes(OpTag::MaxDim), expect);
        // Row-shape identity: byte-identical to the SumDim row for the same
        // attrs (the tag disambiguates via op_name, never via the blob).
        assert_eq!(
            a.to_canonical_bytes(OpTag::MaxDim),
            a.to_canonical_bytes(OpTag::SumDim),
            "MaxDim must share the SumDim reduce-row schema"
        );
    }

    // ---- T9 (Increment C slice 1): matmul role-vector serialize + derive -----
    //
    // The LOCKED reply-3 layout (commit `b64aa1db`; §5 of the recipe-grammar
    // design input): matmul's op_attrs = two per-axis role vectors over
    // { Batch=0, FreeM=1, FreeN=2, ContractedK=3 } (one u8 per axis), `lhs_roles`
    // then `rhs_roles`, each length = operand rank —
    //   body = u32_le(len lhs) ++ lhs_roles ++ u32_le(len rhs) ++ rhs_roles
    // wrapped by carrier (a)'s outer u32_le(body_len) frame. Role vectors encode
    // WHICH axis plays which role, not extents, so GQA (differing-but-divisible
    // batch extents) serializes to identical all-Batch leading roles.

    #[test]
    fn matmul_role_vectors_serialize_the_locked_rank2_golden() {
        // Worked rank-2 example: lhs=[FreeM,ContractedK]=[1,3], rhs=[ContractedK,FreeN]=[3,2].
        //   body = 02000000 | 0103 | 02000000 | 0302   (12 bytes)
        //   full = 0C000000 | body                       (16 bytes)
        //
        // INJECTED (B9): this golden is ALSO the shared CROSS-PRODUCER contract —
        // Baracuda (#68) confirmed the exact bytes and has NO near-term binary
        // arm, so Fuel's serializer is first and this golden IS the contract.
        let a = OpAttrs { lhs_roles: vec![1, 3], rhs_roles: vec![3, 2], ..OpAttrs::default() };
        let golden: Vec<u8> = vec![
            0x0C, 0x00, 0x00, 0x00, // outer u32-LE body length = 12
            0x02, 0x00, 0x00, 0x00, // u32-LE len lhs_roles = 2
            0x01, 0x03, //             lhs_roles = [FreeM, ContractedK]
            0x02, 0x00, 0x00, 0x00, // u32-LE len rhs_roles = 2
            0x03, 0x02, //             rhs_roles = [ContractedK, FreeN]
        ];
        assert_eq!(
            a.to_canonical_bytes(OpTag::MatMul),
            golden,
            "the rank-2 matmul role-vector golden is the shared cross-producer contract (Baracuda #68)"
        );
    }

    #[test]
    fn matmul_empty_roles_stay_the_canonical_zero_body() {
        // Empty roles = the rank-polymorphic recipe form: the body is empty → the
        // single canonical 4-byte zero form. This preserves today's golden
        // (`empty_schema_op_serializes_zero_length`) untouched.
        assert_eq!(OpAttrs::default().to_canonical_bytes(OpTag::MatMul), vec![0, 0, 0, 0]);
    }

    #[test]
    fn matmul_roles_derives_the_canonical_cell() {
        // matmul_roles(lhs_rank, rhs_rank): lhs = [Batch.., FreeM(1), ContractedK(3)];
        // rhs = [Batch.., ContractedK(3), FreeN(2)]. Role POSITIONS, not extents.
        assert_eq!(matmul_roles(2, 2), (vec![1u8, 3], vec![3u8, 2]));
        assert_eq!(matmul_roles(4, 4), (vec![0u8, 0, 1, 3], vec![0u8, 0, 3, 2]));
    }

    // ---- advisory-band drift fixture ----------------------------------------

    #[test]
    fn advisory_band_reference_cases_are_well_formed() {
        // The shared fixture both band impls pin against: non-empty, valid
        // bind spaces, and it exercises every branch of the formula (op-free,
        // single-exact, multi-exact, and transcendental). The VALUES are
        // asserted from the two formula sides (adapter + fuel-dispatch); here
        // we only guard the fixture's own structure so a malformed region can
        // never silently weaken both sides at once.
        let cases = advisory_band_reference_cases();
        assert!(cases.len() >= 8, "fixture should cover the formula's branches");
        for (region, _expected) in &cases {
            let binds = region.bind_indices();
            if let PatternNode::Op { .. } = region {
                // Every op region here is closed over a contiguous [0, n) bind
                // space (the fixtures are concrete recipe regions).
                for (i, b) in binds.iter().enumerate() {
                    assert_eq!(*b as usize, i, "bind indices must be [0, n)");
                }
            }
        }
        // Branch coverage: at least one op-free (expected None), one exact
        // multi-op (Some), and one transcendental band value present.
        assert!(cases.iter().any(|(r, e)| matches!(r, PatternNode::Bind { .. }) && e.is_none()));
        assert!(cases.iter().any(|(_, e)| *e == Some(1)));
        assert!(cases.iter().any(|(_, e)| *e == Some(4)));
        assert!(cases.iter().any(|(_, e)| *e == Some(8)));
    }
}
