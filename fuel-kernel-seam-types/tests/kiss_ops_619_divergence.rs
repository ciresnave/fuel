// SPDX-License-Identifier: MIT OR Apache-2.0
//! GAP-305: the carrier-set ops this crate also encodes, measured against
//! KISS-Ops §6.19.3's schemas -- and DIVERGING from every one of them.
//!
//! ⚠️ THIS IS NOT AN INTEROP TEST AND MUST NOT BE READ AS ONE. There is no
//! interop on this carrier to test. `OpAttrs::to_canonical_bytes` is the #67
//! NODE-ENVELOPE (carrier (a) of `lib.rs`'s three-carrier pin): u32-LE outer
//! length, "payload verbatim, no-parse-inside (§6.19-0010)". A receiver that
//! does not parse the body cannot be byte-comparing it. The named
//! counterparty also emits no such blob -- 0 encode / serialize / to_bytes /
//! canonical sites on `OpAttrs`, control 46 -- but that was measured BY
//! BARACUDA'S LANE and is repeated here on their authority, not re-measured.
//! The no-parse-inside argument above stands without it.
//!
//! WHAT IT DOES TEST: that OUR encoder still stands where we said it stands
//! relative to the schema we BORROWED. Every expected-KISS vector below is
//! derived from the CLAUSE TEXT of KISS-Ops §6.19.3 -- field order, pinned
//! width, resolved default -- and never from this crate's output, which would
//! rebuild a relative oracle with extra steps.
//!
//! IT REDDENS IN BOTH DIRECTIONS, ON PURPOSE:
//!   * if Fuel's encoding moves, the `expect_fuel` vectors fail;
//!   * if Fuel ever becomes §6.19-conformant, the `assert_ne!`s fail and this
//!     file must be rewritten as a conformance test. A divergence that closes
//!     silently is as bad as one that opens silently.
//!
//! Measured against KISS `origin/main` `2b673e0a`, 2026-09-09. §6.19-0003 is
//! scoped "for this op-set version" and this crate pins no KISS-Ops version --
//! see the GAP row. If a version is ever pinned, re-derive these vectors.

use fuel_kernel_seam_types::{OpAttrs, OpTag};

/// Strip the 4-byte u32-LE envelope and hand back the body.
///
/// The length check is the positive control for every test here: it proves the
/// encoder ran and produced a well-formed envelope, so no `assert_ne!` below
/// can pass by comparing against nothing.
fn body(attrs: &OpAttrs, op: OpTag) -> Vec<u8> {
    let blob = attrs
        .to_canonical_bytes(op)
        .unwrap_or_else(|e| panic!("{op:?} declined: {e}"));
    assert!(
        blob.len() >= 4,
        "{op:?}: envelope shorter than its u32 prefix"
    );
    let declared = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize;
    assert_eq!(
        declared,
        blob.len() - 4,
        "{op:?}: §6.19-0010 definite length -- the u32 LE prefix must equal the body length"
    );
    blob[4..].to_vec()
}

// KISS-OPS-6.19-0027 -- gather:
//   "the `gather` OpAttrs blob MUST be exactly `axis` (`u8` ...) then
//    `oob_policy` (enum `u8`, default `1` skip) then `index_operand` (`u8` ...)
//    then `index_dtype` (enum `u8`, mandatory), in that order."
//
// Derived from that sentence alone: FOUR bytes. `axis` = 2 for this vector;
// `oob_policy` takes its RESOLVED default 1, which §6.19-0005 forbids eliding.
const KISS_GATHER_AXIS2: &[u8] = &[2, 1, 0, 0];

#[test]
fn gather_diverges_from_kiss_ops_6_19_0027() {
    let attrs = OpAttrs {
        axis: Some(2),
        ..Default::default()
    };
    let fuel = body(&attrs, OpTag::Gather);

    assert_eq!(
        fuel,
        2i64.to_le_bytes().to_vec(),
        "Fuel's gather row is a single i64 axis; if this moved, the divergence \
         below is being measured against the wrong baseline"
    );
    assert_ne!(
        fuel, KISS_GATHER_AXIS2,
        "gather now MATCHES §6.19-0027. If Fuel became conformant this file is \
         obsolete and must be rewritten as a conformance test -- do not delete \
         the assertion to make it pass."
    );
    assert_eq!(
        (fuel.len(), KISS_GATHER_AXIS2.len()),
        (8, 4),
        "the field-set divergence: §6.19 carries oob_policy, index_operand and \
         index_dtype that Fuel does not"
    );
}

// KISS-OPS-6.19-0034 -- index_select / scatter_add (and embedding):
//   "MUST each be exactly `axis` (`u8` ...) then `index_operand` (`u8`) then
//    `index_dtype` (enum `u8`), in that order" -- THREE bytes.
const KISS_INDEX_TRIPLE_AXIS0: &[u8] = &[0, 0, 0];

#[test]
fn index_select_and_scatter_add_diverge_from_kiss_ops_6_19_0034() {
    for op in [OpTag::IndexSelect, OpTag::ScatterAdd] {
        let attrs = OpAttrs {
            axis: Some(0),
            ..Default::default()
        };
        let fuel = body(&attrs, op);
        assert_eq!(fuel, 0i64.to_le_bytes().to_vec(), "{op:?} row moved");
        assert_ne!(
            fuel, KISS_INDEX_TRIPLE_AXIS0,
            "{op:?} now matches §6.19-0034"
        );
    }
}

// KISS-OPS-6.19-0025 -- reduce:
//   "MUST be exactly `monoid` (`u8`, mandatory non-zero ordinal) then
//    `reduce_axes` (`u16` LE ...) then `keepdim` (bool `u8`, fixed `1`) then
//    `accumulator` (`u8`) then `math_precision` (`u8`)" -- SIX bytes.
const KISS_REDUCE_LEN: usize = 6;

#[test]
fn dim_reduce_diverges_from_kiss_ops_6_19_0025() {
    for op in [OpTag::SumDim, OpTag::MaxDim, OpTag::MeanDim] {
        let attrs = OpAttrs {
            axis: Some(1),
            ..Default::default()
        };
        let fuel = body(&attrs, op);

        let mut expect_fuel = 1i64.to_le_bytes().to_vec();
        expect_fuel.push(0);
        assert_eq!(fuel, expect_fuel, "{op:?} row moved");

        assert_ne!(
            fuel.len(),
            KISS_REDUCE_LEN,
            "{op:?} body is now §6.19-0025's length"
        );
        // ⚠️ A VALUE divergence sitting inside the field-set divergence, and it
        // is invisible in a length comparison: §6.19-0025 fixes keepdim at 1,
        // Fuel emits 0 because these tags remove the reduced dim.
        assert_eq!(
            fuel[8], 0,
            "{op:?} emits keepdim=0 where §6.19-0025 fixes it at 1, so the rows \
             disagree on a field they BOTH carry"
        );
    }
}

// KISS-OPS-6.19-0026 -- prefix_scan (Fuel's CumSum): `monoid` then
// `reduce_axes` (u16 LE) then `exclusivity` then `accumulator` then
// `math_precision` -- SIX bytes.
#[test]
fn cumsum_diverges_from_kiss_ops_6_19_0026() {
    let attrs = OpAttrs {
        axis: Some(1),
        ..Default::default()
    };
    let fuel = body(&attrs, OpTag::CumSum);
    let mut expect_fuel = 1i64.to_le_bytes().to_vec();
    expect_fuel.push(0);
    assert_eq!(fuel, expect_fuel, "CumSum row moved");
    assert_ne!(fuel.len(), 6, "CumSum body is now §6.19-0026's length");
}

/// ⚠️ THE DIVERGENCE IS NOT ONLY A FIELD SET -- THE ONE FIELD BOTH SIDES CARRY
/// IS A DIFFERENT WIDTH, AND NOTHING RECORDED THAT.
///
/// `canonical.rs`'s scope note describes the gather/scatter divergence as Fuel
/// emitting `{axis}` where §6.19 schemas `{axis, oob_policy, index_operand,
/// index_dtype, scatter_combine}` -- a FIELD-SET difference. But
/// KISS-OPS-6.19-0007 pins "axis-index and operand-index and vector
/// length-prefixes and `u8` enum ordinals as ONE BYTE", and Fuel emits `axis`
/// as an i64.
///
/// So the two rows do not agree even on the single field they share. A reader
/// who adopted only the field-set framing would believe an `{axis}`-only
/// consumer could read Fuel's byte; it would read eight.
#[test]
fn the_shared_axis_field_diverges_on_width_not_only_on_field_set() {
    let attrs = OpAttrs {
        axis: Some(2),
        ..Default::default()
    };
    let fuel = body(&attrs, OpTag::Gather);
    assert_eq!(fuel.len(), 8, "Fuel encodes axis at i64 width");
    assert_eq!(
        KISS_GATHER_AXIS2[0], 2,
        "§6.19-0007 encodes an axis index in ONE byte"
    );
    assert_ne!(
        fuel.len(),
        1,
        "if Fuel narrowed axis to u8 this width divergence closed and the prose \
         describing it must be updated"
    );
}

/// OVER-SCOPE ARM. Every op measured in this file must be IN §6.19-0003's
/// carrier set, or the comparison is theatre -- a non-carrier op has no §6.19
/// row to diverge FROM, so asserting a difference against a schema that does
/// not exist proves nothing.
///
/// The set is written out rather than derived, because it is a claim about
/// KISS's document and not about Fuel's code: if it drifts, that is a spec
/// change someone has to read.
#[test]
fn every_op_measured_here_is_in_the_kiss_carrier_set() {
    const KISS_CARRIER_SET: &[&str] = &[
        "reduce",
        "prefix_scan",
        "gather",
        "scatter",
        "sort_network",
        "reduce_var",
        "reduce_std",
        "softmax",
        "log_softmax",
        "rms_norm",
        "layer_norm",
        "avg_pool",
        "max_pool",
        "im2col",
        "index_select",
        "embedding",
        "scatter_add",
    ];
    // The mapping is Fuel's own, documented in canonical.rs's scope note: the
    // monoid rides op_name for the reduce family, and scatter_combine rides
    // op_name for IndexAdd vs ScatterAdd.
    for (tag, kiss) in [
        (OpTag::Gather, "gather"),
        (OpTag::IndexSelect, "index_select"),
        (OpTag::ScatterAdd, "scatter_add"),
        (OpTag::SumDim, "reduce"),
        (OpTag::MaxDim, "reduce"),
        (OpTag::MeanDim, "reduce"),
        (OpTag::CumSum, "prefix_scan"),
    ] {
        assert!(
            KISS_CARRIER_SET.contains(&kiss),
            "{tag:?} maps to `{kiss}`, which is NOT in §6.19-0003's carrier set"
        );
    }

    // The negative half. `Slice` is the worked example: §6.19-0003 requires a
    // non-carrier op to emit an EMPTY blob, and Fuel emits twenty bytes -- so
    // it has no §6.19 row and must never be measured above.
    let slice = OpAttrs {
        axis: Some(0),
        slice_start: Some(0),
        slice_len: Some(4),
        ..Default::default()
    };
    let emitted = body(&slice, OpTag::Slice);
    assert_eq!(
        emitted.len(),
        20,
        "Slice emits u32 axis + u64 start + u64 len"
    );
    assert!(
        !KISS_CARRIER_SET.contains(&"slice"),
        "if `slice` entered the carrier set it acquired a §6.19 row and belongs \
         in the measured set above"
    );
}
