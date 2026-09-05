// SPDX-License-Identifier: MIT OR Apache-2.0
//! Carrier (a): the canonical byte serialization of [`OpAttrs`] for an
//! [`OpTag`], and its typed decline.
//!
//! ⚠️ WHY THIS IS ITS OWN FILE. Codacy's metric engine counts COMMENTS as lines
//! of code — measured: the pre-GAP-287 `to_canonical_bytes` was 114 lines / 74
//! code / 40 comments and scored 112. Its per-arm rationale (which fields
//! decline, which keep their default, and why each) is the substance of
//! GAP-287, so the function is over the limit BECAUSE it is documented.
//!
//! Codacy's only in-repo suppression scope is a FILE GLOB — there is no
//! per-function or inline mechanism (checked: `.codacy.yml` supports
//! `exclude_paths` with Java glob syntax; ignoring individual issues is a web-UI
//! action). Excluding `lib.rs` would silently exempt every other function in a
//! 1,700-line file. So the serializer moved into a file whose scope IS the thing
//! being exempted, rather than the exemption being widened to fit the code.
//!
//! Same principle as the project rule "when a gate fires on your own tooling,
//! MOVE THE TOOL rather than teach the gate an exception": an exemption is
//! permanent and invisible in the gate's claim; a moved file is neither.

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
    /// field from a genuine zero (e.g. `axis: None` vs `Some(0)`). Harmless
    /// today — there is no decoder; this is a forward-serialization only, and an
    /// op that reaches a given arm always has the field set (`op_to_attrs` /
    /// `tag_to_op` guarantee it). A future decoder must not round-trip `None`.
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
    fn canonical_body(&self, op: OpTag) -> Result<Vec<u8>, UnresolvedAttr> {
        // The four ACKED source-op LEAF arms are a separate table: they are
        // WIRE-ONLY tokens (`op_to_tag` emits none of them, `tag_to_op` has no
        // arm for any of them), so they share a decline rule the arms below do
        // not. Tried first so the match below is exactly the graph-projected
        // schema and nothing else.
        if let Some(leaf) = self.leaf_body(op)? {
            return Ok(leaf);
        }
        use OpTag as T;
        let req = |v: Option<i64>, field| v.ok_or(UnresolvedAttr { op, field });
        let req_u64 = |v: Option<u64>, field| v.ok_or(UnresolvedAttr { op, field });
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
            T::AddScalar | T::MulScalar | T::Clamp | T::PowI => {
                put_f64_list(&mut body, &self.scalars);
            }
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
            // A guard rather than a nested `if`: MatMul with BOTH role vectors
            // empty is the rank-polymorphic implicit form, whose body is empty,
            // so it falls to the empty-schema arm below and emits the canonical
            // `[00,00,00,00]`. Same bytes as before, one less branch.
            T::MatMul if !self.lhs_roles.is_empty() || !self.rhs_roles.is_empty() => {
                put_u8_list(&mut body, &self.lhs_roles);
                put_u8_list(&mut body, &self.rhs_roles);
            }
            // Empty-schema ops (elementwise, comparison, Where, scalar
            // reductions, log-softmax, ...) and any tag added later: zero-length.
            _ => {}
        }
        Ok(body)
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
