// SPDX-License-Identifier: MIT OR Apache-2.0
//! Carrier (a): the canonical byte serialization of [`OpAttrs`] for an
//! [`OpTag`], and its typed decline.
//!
//! WHY THIS IS ITS OWN FILE: the three methods, the typed decline, and the
//! little-endian writers are one unit — every use of `put_*` outside the
//! serializer is one writer calling another, checked rather than assumed.
//!
//! ⚠️ A CLAIM THAT WAS HERE AND WAS FALSE, recorded because it was acted on.
//! This header asserted that "Codacy's metric engine counts COMMENTS as lines
//! of code", inferred from the pre-GAP-287 `to_canonical_bytes` being 114 lines
//! / 74 code / 40 comments and scoring 112. **Refuted, on this file:** Codacy
//! reports `canonical_body` at 83 lines of code and it has exactly 83
//! non-comment lines; `leaf_body` at 39 is not flagged at all. Codacy counts
//! CODE and the limit is 50.
//!
//! The reasoning error is worth more than the fact: 112 matches neither 74 nor
//! 114, and the closer candidate was treated as confirmed. Ruling out one
//! hypothesis narrows the space; it does not populate it. The functions below
//! are split because they are genuinely large, not because documentation is
//! penalised — that story was never true.

use crate::{OpAttrs, OpTag};

/// A typed decline from [`OpAttrs::to_canonical_bytes`]: the arm for `op` reads
/// `field`, and `field` is unset.
///
/// GAP-287. The serializer used to substitute `unwrap_or` defaults here, which
/// made `Gather` on axis 2 and on axis 0 emit IDENTICAL bytes. KISS-OPS-6.19
/// requires every emitted field to be "already RESOLVED to its EFFECTIVE VALUE"
/// and has no "absent" state, so an unset required attr has no representation
/// on this wire -- it is a producer error, not a value.
///
/// This is not a new policy for this seam: `fuel_graph::runtime_fused::tag_to_op`,
/// the DECODE direction this serializer round-trips against, already declines an
/// unset required attr on SIXTEEN fields and names the pattern in its own
/// comments ("an honest miss (unset required attr)"). The encoder was the side
/// that dissented.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnresolvedAttr {
    /// The op whose arm reads the field.
    pub op: OpTag,
    /// The `OpAttrs` field that is unset. A `&'static str` rather than an enum:
    /// the set is exactly "the fields this function reads", which is not a
    /// vocabulary any consumer should match on.
    pub field: &'static str,
}

impl core::fmt::Display for UnresolvedAttr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cannot serialize {:?}: required attr `{}` is unset (the canonical              wire has no absent state, so there is no value to emit)",
            self.op, self.field
        )
    }
}

impl std::error::Error for UnresolvedAttr {}

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

fn put_u32(b: &mut Vec<u8>, x: u32) {
    b.extend_from_slice(&x.to_le_bytes());
}
fn put_u64(b: &mut Vec<u8>, x: u64) {
    b.extend_from_slice(&x.to_le_bytes());
}
fn put_i64(b: &mut Vec<u8>, x: i64) {
    b.extend_from_slice(&x.to_le_bytes());
}
fn put_f64(b: &mut Vec<u8>, x: f64) {
    b.extend_from_slice(&x.to_le_bytes());
}
fn put_str(b: &mut Vec<u8>, s: &str) {
    put_u32(b, s.len() as u32);
    b.extend_from_slice(s.as_bytes());
}
fn put_i64_list(b: &mut Vec<u8>, xs: &[i64]) {
    put_u32(b, xs.len() as u32);
    for &x in xs {
        put_i64(b, x);
    }
}
fn put_u32_list(b: &mut Vec<u8>, xs: &[u32]) {
    put_u32(b, xs.len() as u32);
    for &x in xs {
        put_u32(b, x);
    }
}
fn put_f64_list(b: &mut Vec<u8>, xs: &[f64]) {
    put_u32(b, xs.len() as u32);
    for &x in xs {
        put_f64(b, x);
    }
}
fn put_u8_list(b: &mut Vec<u8>, xs: &[u8]) {
    put_u32(b, xs.len() as u32);
    b.extend_from_slice(xs);
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
    /// field from a genuine zero (e.g. `axis: None` vs `Some(0)`), and for five
    /// (op, field) pairs that collapse is **LOSSY TODAY** - two semantically
    /// distinct ops serialize to identical bytes. Registry: GAP-287.
    ///
    /// This paragraph previously read *"harmless today ... an op that reaches a
    /// given arm always has the field set (`op_to_attrs` / `tag_to_op`
    /// guarantee it)"*. **That was false, and the guarantor's own doc says so**:
    /// `fuel_graph::jit::op_to_attrs` documents itself as *"not exhaustive"* and
    /// leaves `axis` unset for `CumSum`/`IndexSelect`/`Gather`/`IndexAdd`/
    /// `ScatterAdd` (its `_ => {}` arm). So `Gather` on axis 2 and on axis 0
    /// emit the same bytes. KISS-OPS-6.19 makes `axis` MANDATORY with no
    /// default, so `unwrap_or(0)` fabricates a value the schema does not permit.
    ///
    /// `keepdim` is also set by **nothing in the repository**, and that one is
    /// NOT a defect - the distinction worth keeping. Every tag on that arm
    /// removes the reduced dim (`Op::SumDim`/`MaxDim`/`MeanDim` say so verbatim;
    /// `CumSum` is "same shape as input"), so `false` is the true value.
    /// keepdim=TRUE is `Op::ReduceSumTo(Shape)`: a different tag, on the
    /// shape-target arm, with no `unwrap_or` at all. **A field being unset is
    /// only a defect if the default CAN differ from the truth** - "never set"
    /// has several causes and only one of them is this bug.
    ///
    /// `const_bits`, `slot_index`, `scan_role` and `scan_index` are reachable
    /// only from hand-built regions (`op_to_tag` emits none of the four leaf
    /// tags); `scan_*` are set together by both producers today, guaranteed by
    /// nothing. Note `scan_role.unwrap_or(SCAN_ROLE_CARRY)` is the only default
    /// here that names a *semantic* value rather than a neutral zero: an unset
    /// role does not decay to "absent", it asserts "carry".
    ///
    /// The old note also said *"a future decoder must not round-trip `None`"*.
    /// **A decoder cannot recover what this encoder already discarded** - the
    /// mitigation has to live on this side of the wire. And *"harmless, there is
    /// no decoder"* was scoped to Fuel's own repo while this format exists to
    /// leave it - the decoder is the counterparty's by design, i.e. the claim
    /// was scoped to the one place the hazard cannot occur.
    ///
    /// On what checks this format: exactly one conformance fixture exists,
    /// `matmul_role_vectors_serialize_the_locked_rank2_golden`, and Baracuda
    /// "has NO near-term binary arm" (its own comment), so there is no second
    /// encoder. That golden covers the `MatMul` arm - which emits `lhs_roles`/
    /// `rhs_roles` directly and contains **no `unwrap_or` at all**. The single
    /// fixture this format has is aimed at an arm that cannot exhibit this
    /// defect, which is why six lossy collapses were found by audit rather
    /// than by a failing test.
    ///
    /// ⚠️ **THIS CRATE'S TWO DIRECTIONS DISAGREE, AND `axis` IS ONE OF THEM.**
    /// `fuel_graph::runtime_fused::tag_to_op` - the DECODE direction this
    /// serializer round-trips against - treats an unset required attr as a hard
    /// decline: **24 `attrs.<field>?` declines over 11 fields**, plus 5 explicit
    /// `return None`, and its own comments name the policy ("an honest miss
    /// (unset required attr)"). Seven fields appear on BOTH sides:
    ///
    /// ```text
    /// axis  pad_mode  roll_shift  scan_index  scan_role  slice_len  slice_start
    /// ```
    ///
    /// For every one of them the decoder REFUSES what this encoder INVENTS.
    /// Six are latent (`op_to_attrs` does set them, so the default is currently
    /// unreachable); `axis` is LIVE, because the five ops above are the ones it
    /// leaves unprojected. So making the encoder decline is not a new design -
    /// it is this codebase's own settled answer to this exact condition, and
    /// the encoder is the side that dissents.
    ///
    /// Gated by the collapse table in `fuel-graph/src/jit.rs` tests, which pins
    /// exactly which (op, field) pairs are lossy so that fixing one is visible.
    /// Fixing the encoder is a wire-format change and needs the external ask.
    /// Serialize `self` for `op` as carrier (a): `u32_le(body_len) ++ body`.
    ///
    /// The FRAME is separated from the per-op BODY here only because they are
    /// different things -- the envelope is fixed for every tag, the body is the
    /// schema table. The table itself is deliberately NOT split further: it is
    /// one match over the wire schema, and a reader checking Fuel against
    /// KISS-OPS-6.19 needs to see all of it at once.
    pub fn to_canonical_bytes(&self, op: OpTag) -> Result<Vec<u8>, UnresolvedAttr> {
        let body = self.canonical_body(op)?;
        let mut out = (body.len() as u32).to_le_bytes().to_vec();
        out.extend_from_slice(&body);
        Ok(out)
    }

    /// The per-op body, unframed. One arm per SCHEMA FAMILY rather than per
    /// tag: tags sharing a row shape share an arm, which is why `MaxDim` and
    /// `SumDim` are byte-identical for equal attrs (the monoid rides `op_name`).
    ///
    /// # ⚠️ A STANDING CODACY FINDING ON THIS FUNCTION IS ACCEPTED AND PERMANENT
    ///
    /// Codacy reports `Method canonical_body has N lines of code (limit is 50)`
    /// and a cyclomatic complexity over 12. **That finding is correct, it is not
    /// going to be fixed, and it will never clear.** It is recorded here so a
    /// reader does not mistake a permanent red for unfinished work: the first
    /// thing anyone learns from a check that cannot go green is to stop reading
    /// it, and an accepted finding with no author reads as one somebody meant to
    /// get back to.
    ///
    /// **WHY IT CANNOT GO DOWN.** Measured at this commit: 68 lines of code, 6
    /// match arms, **17 distinct `OpTag` variants**. Each variant carries a
    /// DIFFERENT WIRE ROW SCHEMA under KISS-OPS-§6.19 — `Slice` is
    /// `{axis, start, len}`, the single-axis family is `{axis}`, the reduces are
    /// `{axis, keepdim}`, `Cast` is a length-prefixed string, `Pad` is a list
    /// plus mode plus value. **Rows that differ on the wire cannot be collapsed
    /// into one arm.** The count is set by the op vocabulary, not by style, so
    /// it goes UP when the vocabulary grows and never down.
    ///
    /// Re-derive it by counting `T::` variants in the match and comparing them
    /// against §6.19's row schemas. If a future reader finds two arms with the
    /// SAME row shape, that is a real finding and this note is wrong.
    ///
    /// **WHAT WAS ALREADY SPLIT OUT, so this is not an un-attempted excuse:**
    /// `to_canonical_bytes` (the envelope, 6), [`Self::list_body`] (26, the arms
    /// that cannot decline), [`Self::leaf_body`] (39, the wire-only tags). The
    /// split axis was *does this arm read a field that can be unset* — the
    /// GAP-287 question itself. `leaf_body` at 39 passes, which is what shows
    /// the mechanism works rather than that the limit is unreachable.
    ///
    /// ⚠️ **AND WHAT THIS NOTE DOES *NOT* CLAIM.** An earlier draft justified the
    /// finding as "exhaustiveness is load-bearing here — no wildcard is
    /// permitted". **That is false and was measured false before it was
    /// written:** `OpTag` is `#[non_exhaustive]`, this match ends in `_ => {}`
    /// (see below), and there is no `EXHAUSTIVE-BY-DESIGN` marker in this crate.
    /// Those things are true elsewhere in Fuel and not here. A justification that
    /// sounds checkable and fails when checked is worse than none.
    fn canonical_body(&self, op: OpTag) -> Result<Vec<u8>, UnresolvedAttr> {
        // Three emitters, one per SCHEMA FAMILY, and this function only
        // DISPATCHES. `leaf_body` and `list_body` were split out first; leaving
        // the declining arms inline was the asymmetry, not a decision.
        if let Some(leaf) = self.leaf_body(op)? {
            return Ok(leaf);
        }
        if let Some(b) = self.list_body(op) {
            return Ok(b);
        }
        self.declining_body(op)
    }

    /// The arms that CAN decline: every one reads an `Option` field that
    /// KISS-OPS-6.19 requires to be resolved, so every one is a branch.
    ///
    /// Split from [`Self::canonical_body`] to complete the one-emitter-per-family
    /// shape rather than to hit a threshold: `list_body` (infallible) and
    /// `leaf_body` (wire-only) were already separate, and these were the
    /// remainder left inline.
    ///
    /// # ⚠️ A STANDING CODACY COMPLEXITY FINDING HERE IS ACCEPTED AND PERMANENT
    ///
    /// Codacy reports a cyclomatic complexity over its limit of 12 on this
    /// function. **The finding is correct and it is not going to be fixed.**
    /// Recorded so a permanent red is not mistaken for unfinished work.
    ///
    /// **(1) THE BRANCH COUNT IS SPEC-DRIVEN — and the margin is ONE.** Measured
    /// at this commit: cyclomatic **13** against a limit of 12, from ten
    /// required-field checks across six tag arms, one per field KISS-OPS-6.19
    /// declares mandatory-with-no-default. It goes UP when 6.19 adds a mandatory
    /// field.
    ///
    /// ⚠️ **This ground is stated narrowly on purpose.** An earlier draft argued
    /// the metric was *unreachable* for this shape — that was written against an
    /// estimated complexity of ~19 and a pre-extraction reading of 16, and the
    /// measured value is 13. **"The metric cannot be reached" and "we are one
    /// branch over" are different claims, and only the second is true.** So (1)
    /// explains where the branches come from; it does not by itself carry the
    /// decision. Grounds (2) and (3) do.
    ///
    /// **(2) THE OBVIOUS REFACTOR WOULD DESTROY A GUARANTEE.** Driving the
    /// required-field checks from a table would cut the branch count and
    /// dissolve the per-arm `match`. ⚠️ **GAP-290 makes that `match` structure
    /// load-bearing: with the wildcard gone, a new `OpTag` is an `E0004`.** *(At
    /// THIS commit the wildcard is still present and GAP-290 is the follow-on
    /// change — so this reason becomes live then, and is written now because it
    /// is the one a future tidier most needs and least expects.)*
    ///
    /// **(3) THE REMAINING CUT WOULD BE INVENTED.** The next available split is
    /// axis-bearing vs value-bearing arms. It was declined and re-checked rather
    /// than re-decided: this file's comments name each arm's schema
    /// INDIVIDUALLY (*"Single-axis ops (dim rides `axis`)"*, *"Cast:
    /// length-prefixed dtype name"*) and draw no such grouping. Inventing one to
    /// satisfy a metric is fitting the code to the instrument, which this repo
    /// already refuses elsewhere in writing.
    ///
    /// **What WAS done, so this is not an un-attempted excuse:** the emitters
    /// were split three ways and `canonical_body` reduced to a dispatcher. That
    /// **moved** the complexity here rather than reducing it — reported as such,
    /// because the split was justified before the metric and not by it.
    fn declining_body(&self, op: OpTag) -> Result<Vec<u8>, UnresolvedAttr> {
        use OpTag as T;
        let req = |v: Option<i64>, field| v.ok_or(UnresolvedAttr { op, field });
        let req_u64 = |v: Option<u64>, field| v.ok_or(UnresolvedAttr { op, field });
        let mut body: Vec<u8> = Vec::new();
        match op {
            // Shape-target ops: the logical output shape (Iota's len rides it).
            T::Slice => {
                put_u32(&mut body, req(self.axis, "axis")? as u32);
                put_u64(&mut body, req_u64(self.slice_start, "slice_start")?);
                put_u64(&mut body, req_u64(self.slice_len, "slice_len")?);
            }
            // Single-axis ops (dim rides `axis`).
            T::Concat
            | T::Flip
            | T::Triu
            | T::Tril
            | T::IndexSelect
            | T::Gather
            | T::IndexAdd
            | T::ScatterAdd => {
                put_i64(&mut body, req(self.axis, "axis")?);
            }
            // Roll: axis(i64) + shift(i64).
            T::Roll => {
                put_i64(&mut body, req(self.axis, "axis")?);
                put_i64(&mut body, req(self.roll_shift, "roll_shift")?);
            }
            // Dim reductions + cumsum: axis(i64) + keepdim(u8). The monoid
            // rides op_name (distinct SumDim/MaxDim/MeanDim tags), so every
            // reduce tag shares this one row schema.
            //
            // `keepdim` KEEPS its default and that is deliberate: every tag on
            // this arm removes the reduced dim (`CumSum` does not reduce at
            // all), so `false` is the true value, not a fabrication. A
            // keepdim=TRUE reduce is `ReduceSumTo`/`ReduceMaxTo` -- a different
            // tag on the shape-target arm above. `tag_to_op` reads no `keepdim`
            // for these tags, so declining here would refuse input the decoder
            // accepts. See `keepdim_is_a_dead_field_not_a_lossy_collapse_gap287`.
            T::SumDim | T::MaxDim | T::MeanDim | T::CumSum => {
                put_i64(&mut body, req(self.axis, "axis")?);
                body.push(self.keepdim.unwrap_or(false) as u8);
            }
            // Cast: length-prefixed dtype name. `tag_to_op` DECLINES an unset
            // `cast_dtype` here (`attrs.cast_dtype.as_deref()?`), so emitting
            // `""` would produce a blob the decoder must reject -- a fabrication
            // that does not even round-trip.
            T::Cast => put_str(
                &mut body,
                self.cast_dtype.as_deref().ok_or(UnresolvedAttr {
                    op,
                    field: "cast_dtype",
                })?,
            ),
            // Pad: amounts (count + (before:u64, after:u64) each) + mode(u8) + value(f64).
            //
            // `pad_mode` is required (`tag_to_op` declines it). `pad_value`
            // KEEPS its `0.0` default because the decoder defaults it
            // identically (`attrs.pad_value.unwrap_or(0.0)`), so the two
            // directions agree and the round-trip is exact.
            T::Pad => {
                put_u32(&mut body, self.pad_amounts.len() as u32);
                for &(before, after) in &self.pad_amounts {
                    put_u64(&mut body, before);
                    put_u64(&mut body, after);
                }
                body.push(self.pad_mode.ok_or(UnresolvedAttr {
                    op,
                    field: "pad_mode",
                })?);
                put_f64(&mut body, self.pad_value.unwrap_or(0.0));
            }
            // Scalar-param ops: the scalar list.
            // MaskedFill: scalar value(s) + value dtype name.
            //
            // Unlike `Cast`, `tag_to_op` DEFAULTS this one (`None => F32`). But
            // the old encoder wrote `""`, which is not a dtype and which
            // `DType::from_str` rejects -- so the emitted blob did not
            // round-trip through the decoder's own default. Declining is the
            // only choice that agrees with the decoder on every input it can
            // actually be handed.
            T::MaskedFill => {
                put_f64_list(&mut body, &self.scalars);
                put_str(
                    &mut body,
                    self.cast_dtype.as_deref().ok_or(UnresolvedAttr {
                        op,
                        field: "cast_dtype",
                    })?,
                );
            }
            // MatMul: the LOCKED role-vector contraction descriptor (§5,
            // reply-3) -- `u32_le(len lhs) ++ lhs_roles ++ u32_le(len rhs) ++
            // rhs_roles`, u8 roles, lhs-then-rhs. Both empty => the empty body
            // (the canonical `[00,00,00,00]` implicit form; recipes keep matmul
            // rank-polymorphic). The rank-2 golden is the shared cross-producer
            // fixture (Baracuda #68).
            // Empty-schema ops (elementwise, comparison, Where, scalar
            // reductions, log-softmax, ...) and any tag added later: zero-length.
            _ => {}
        }
        Ok(body)
    }

    /// The arms that CANNOT decline: they emit a length-prefixed list or a role
    /// vector, and read no `Option` field.
    ///
    /// Returns `Option<Vec<u8>>` rather than `Result<Option<..>>` deliberately.
    /// There is no error channel because there is no way to fail, and the
    /// signature is the cheapest possible statement of that -- a guard whose
    /// sentence is true. `None` means "not one of these arms", so it is total
    /// over `OpTag` and the caller needs no unreachable case.
    fn list_body(&self, op: OpTag) -> Option<Vec<u8>> {
        use OpTag as T;
        let mut body: Vec<u8> = Vec::new();
        match op {
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
            T::AddScalar | T::MulScalar | T::Clamp | T::PowI => {
                put_f64_list(&mut body, &self.scalars);
            }
            // A guard rather than a nested `if`: MatMul with BOTH role vectors
            // empty is the rank-polymorphic implicit form, whose body is empty,
            // so it falls to the empty-schema arm below and emits the canonical
            // `[00,00,00,00]`. Same bytes as before, one less branch.
            T::MatMul if !self.lhs_roles.is_empty() || !self.rhs_roles.is_empty() => {
                put_u8_list(&mut body, &self.lhs_roles);
                put_u8_list(&mut body, &self.rhs_roles);
            }
            _ => return None,
        }
        Some(body)
    }

    /// The four ACKED source-op LEAF arms (KISS ruling record, "four-leaf-arm
    /// ack", 2026-07-23 -- acked clean, no amendments).
    ///
    /// `Ok(None)` means "not one of the four", so this is TOTAL over `OpTag`
    /// and the caller needs no unreachable arm. They decline on a narrower
    /// ground than every other arm: `tag_to_op` has NO arm for these tags, so
    /// there is no decoder policy to agree with, and an encoder must never emit
    /// a value it was not given. KISS-OPS-6.19 does not specify `const_bits` or
    /// `slot_index` at all. Unreachable today -- `op_to_tag` emits none of them.
    fn leaf_body(&self, op: OpTag) -> Result<Option<Vec<u8>>, UnresolvedAttr> {
        use OpTag as T;
        let mut body: Vec<u8> = Vec::new();
        match op {
            // --- the four ACKED source-op LEAF arms (KISS ruling record,
            // "four-leaf-arm ack", 2026-07-23 -- acked clean, no amendments) ---
            //
            // `tag_to_op` has NO arm for these four tags, so there is no decoder
            // policy to agree with. They decline rather than default, on the
            // narrower ground that an encoder must never emit a value it was
            // not given. KISS-OPS-6.19 does not specify `const_bits` or
            // `slot_index` at all, so there is also no schema default to appeal
            // to. Unreachable today: `op_to_tag` emits none of the four.
            //
            // `const`: u64(bits) -- a DTYPE-AGNOSTIC bit pattern (Q7: the
            // structural DAG carries no dtype). MBZ narrow-dtype rule: a
            // sub-64-bit dtype places its STORAGE bits in the LOW-order bits
            // with the upper bits ZERO (must-be-zero on read) -- producers widen
            // via [`const_bits_narrow`]. A NaN payload is carried verbatim; this
            // serializer never quiets or canonicalizes it.
            T::Const => put_u64(
                &mut body,
                self.const_bits.ok_or(UnresolvedAttr {
                    op,
                    field: "const_bits",
                })?,
            ),
            // `runtime_scalar`: u32(slot_index) -- a DISPATCH-BOUND scalar, a
            // distinct leaf from a baked `const` (an unfilled slot and a baked
            // value are not interchangeable). Defaulting an unfilled slot to
            // slot 0 would silently BIND it, which is the sharpest case for
            // declining.
            T::RuntimeScalar => put_u32(
                &mut body,
                self.slot_index.ok_or(UnresolvedAttr {
                    op,
                    field: "slot_index",
                })?,
            ),
            // `reduced_count`: i64(axis) -- single-axis, byte-identical to the
            // fold row's leading axis field minus `keepdim`, so a resolver
            // reuses the fold's axis-resolution codepath verbatim. Growth to a
            // `reduce_axes` LIST happens ONLY in lockstep with the fold
            // (§6.12-0001), never unilaterally here. `axis` is MANDATORY with no
            // schema default (KISS-OPS-6.19), so it declines like every other
            // axis-bearing arm.
            T::ReducedCount => put_i64(
                &mut body,
                self.axis.ok_or(UnresolvedAttr { op, field: "axis" })?,
            ),
            // `scan_placeholder`: u8(role) ++ u32(index), role 0 = carry,
            // 1 = elem ([`SCAN_ROLE_CARRY`]/[`SCAN_ROLE_ELEM`]).
            //
            // The old default was `unwrap_or(SCAN_ROLE_CARRY)` -- the only
            // default in the whole function that named a SEMANTIC value rather
            // than a neutral zero. An unset role did not decay to "absent", it
            // ASSERTED "carry". `tag_to_op` declines both fields.
            T::ScanPlaceholder => {
                body.push(self.scan_role.ok_or(UnresolvedAttr {
                    op,
                    field: "scan_role",
                })?);
                put_u32(
                    &mut body,
                    self.scan_index.ok_or(UnresolvedAttr {
                        op,
                        field: "scan_index",
                    })?,
                );
            }
            _ => return Ok(None),
        }
        Ok(Some(body))
    }
}
