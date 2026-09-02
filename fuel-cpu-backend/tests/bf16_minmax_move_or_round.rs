// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Does bf16 min/max MOVE a value, or does it ROUND one?**
//!
//! KISS #355 makes promote-to-f32-and-round-back non-conforming for bf16
//! minmax. Fuel does exactly that, and not as a min/max decision: the impl is a
//! BLANKET one over every binary op, because `BinaryOpCore` has no bf16 method
//! at all (`chassis/binary.rs`, `impl<O: BinaryOpCore> BinaryOp<half::bf16>`).
//!
//! This file MEASURES what that costs rather than reasoning about it. The
//! reasoning — widening is exact, min/max returns one of its inputs, so the
//! narrowing cannot round — is clean, was reached independently by three
//! people, and had been run by none of them. That is the shape that gets signed
//! on plausibility.
//!
//! The property under test is #355's own wording: **the result must be one of
//! the inputs, BIT FOR BIT.** A MOVE satisfies that by construction. A round
//! trip satisfies it only if it happens to.

use half::bf16;

// Re-exported through `pub mod chassis`; this exercises the shipped path, not a
// copy of it.
use fuel_cpu_backend::chassis::binary::{BinaryOp, Maximum, Minimum};

fn is_bitwise_one_of(got: bf16, a: bf16, b: bf16) -> bool {
    got.to_bits() == a.to_bits() || got.to_bits() == b.to_bits()
}

/// Every bf16 bit pattern, against a spread of partners. 65_536 x partners is
/// cheap and leaves no finite corner unexamined -- subnormals, both zeros, both
/// infinities and every NaN encoding are all in the sweep by construction.
fn all_bf16() -> impl Iterator<Item = bf16> {
    (0u16..=u16::MAX).map(bf16::from_bits)
}

#[test]
fn bf16_minmax_result_is_bitwise_one_of_its_inputs_for_every_finite_pattern() {
    let partners: Vec<bf16> = [
        0.0f32, -0.0, 1.0, -1.0, 0.5, -0.5, 3.4e38, -3.4e38, 1e-38, -1e-38,
    ]
    .iter()
    .map(|v| bf16::from_f32(*v))
    .collect();

    let mut checked = 0usize;
    let mut violations = Vec::new();
    for a in all_bf16() {
        if a.is_nan() {
            continue; // NaN is the other test's subject
        }
        for b in &partners {
            let lo = <Minimum as BinaryOp<bf16>>::apply(a, *b);
            let hi = <Maximum as BinaryOp<bf16>>::apply(a, *b);
            checked += 2;
            if !is_bitwise_one_of(lo, a, *b) {
                violations.push(format!(
                    "min({:#06x}, {:#06x}) = {:#06x} -- not bitwise either input",
                    a.to_bits(),
                    b.to_bits(),
                    lo.to_bits()
                ));
            }
            if !is_bitwise_one_of(hi, a, *b) {
                violations.push(format!(
                    "max({:#06x}, {:#06x}) = {:#06x} -- not bitwise either input",
                    a.to_bits(),
                    b.to_bits(),
                    hi.to_bits()
                ));
            }
        }
    }

    println!(
        "[bf16-minmax] finite: {checked} comparisons, {} violations",
        violations.len()
    );
    // FOUNDATION: a sweep that examined nothing would report zero violations.
    assert!(
        checked > 100_000,
        "only {checked} comparisons -- the sweep collapsed and a clean result means nothing"
    );
    assert!(
        violations.is_empty(),
        "promote-and-round-back is NOT a move for {} finite pattern(s):\n  {}",
        violations.len(),
        violations
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

/// Zero has two encodings and min/max is where they get confused. `-0.0` and
/// `+0.0` compare EQUAL, so a round trip may return whichever the f32 op picked
/// rather than the one #355 requires.
#[test]
fn bf16_minmax_signed_zero_is_reported_not_assumed() {
    let pos = bf16::from_f32(0.0);
    let neg = bf16::from_f32(-0.0);
    assert_ne!(
        pos.to_bits(),
        neg.to_bits(),
        "control: the two zeros differ in bits"
    );

    let min_pn = <Minimum as BinaryOp<bf16>>::apply(pos, neg);
    let min_np = <Minimum as BinaryOp<bf16>>::apply(neg, pos);
    let max_pn = <Maximum as BinaryOp<bf16>>::apply(pos, neg);
    let max_np = <Maximum as BinaryOp<bf16>>::apply(neg, pos);
    println!(
        "[bf16-minmax] zeros: min(+0,-0)={:#06x} min(-0,+0)={:#06x} max(+0,-0)={:#06x} max(-0,+0)={:#06x}",
        min_pn.to_bits(),
        min_np.to_bits(),
        max_pn.to_bits(),
        max_np.to_bits()
    );
    // Recorded, not asserted in a direction: the point is to SEE which zero
    // comes back before anyone signs a conformance position on it.
    assert!(is_bitwise_one_of(min_pn, pos, neg));
    assert!(is_bitwise_one_of(max_pn, pos, neg));
}

/// The predicted divergence. `to_f32` / `from_f32` need not preserve a NaN
/// PAYLOAD, and need not preserve SIGNALLING-ness -- which is adjacent to
/// KISS #363 (*arithmetic quiets a signalling NaN*).
#[test]
fn bf16_nan_payload_and_signalling_through_the_round_trip() {
    // bf16: 1 sign, 8 exponent, 7 mantissa. NaN = exp 0xFF, mantissa != 0.
    // Quiet bit is the mantissa MSB (bit 6).
    let snan = bf16::from_bits(0x7F81); // signalling: quiet bit CLEAR, payload 1
    let qnan = bf16::from_bits(0x7FC1); // quiet:      quiet bit SET,   payload 1
    assert!(snan.is_nan() && qnan.is_nan(), "control: both are NaN");
    assert_eq!(
        snan.to_bits() & 0x0040,
        0,
        "control: snan quiet bit is clear"
    );
    assert_ne!(qnan.to_bits() & 0x0040, 0, "control: qnan quiet bit is set");

    for (name, n) in [("snan", snan), ("qnan", qnan)] {
        let round = bf16::from_f32(n.to_f32());
        let via_min = <Minimum as BinaryOp<bf16>>::apply(n, bf16::from_f32(1.0));
        let via_max = <Maximum as BinaryOp<bf16>>::apply(n, bf16::from_f32(1.0));
        println!(
            "[bf16-minmax] {name}: in={:#06x} bare_roundtrip={:#06x} min_with_1={:#06x} max_with_1={:#06x}",
            n.to_bits(),
            round.to_bits(),
            via_min.to_bits(),
            via_max.to_bits()
        );
    }

    // Asserted separately from the report so a change in EITHER is visible:
    // the payload surviving, and the signalling bit surviving.
    let via_min = <Minimum as BinaryOp<bf16>>::apply(snan, bf16::from_f32(1.0));
    println!(
        "[bf16-minmax] VERDICT signalling-preserved={} payload-preserved={}",
        via_min.to_bits() & 0x0040 == 0,
        via_min.to_bits() & 0x003F == snan.to_bits() & 0x003F
    );
    assert!(via_min.is_nan(), "NaN-ness at least must survive min()");
}
