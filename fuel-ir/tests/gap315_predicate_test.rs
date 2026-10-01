// SPDX-License-Identifier: MIT OR Apache-2.0
//! GAP-315 FALSIFYING EXPERIMENT — the `stride_dims*` arm.
//!
//! ⚠️ PREDICTIONS WERE REGISTERED BEFORE THIS RAN. This arm's prediction:
//!
//!   (C) `stride_dims2` with a NEGATIVE stride -> (d) SILENT-WRONG.
//!       Reason from the predicate: the LENGTH check above the guard is a real
//!       `if … return Err` and survives release, but the `debug_assert` guards
//!       NON-NEGATIVITY and the next line is `Ok((stride[0] as usize, …))`.
//!       A negative `isize as usize` WRAPS. Nothing reconciles the sign, so the
//!       function returns `Ok` carrying a garbage extent.
//!       Expected arm B: `Ok((18446744073709551615, …))`.
//!
//! THE PREDICATE UNDER TEST (refined after a token classifier failed its own
//! control — it could not recover `conv2d_direct`, a MEASURED (d)):
//!   (d)  the code proceeds on an extent derived from a source OTHER than the
//!        asserted quantity, and NOTHING LATER COMPARES THEM.
//!   (c)  it indexes using the asserted quantity and overruns; bounds check fires.
//!   (a1) something later RECONCILES the two and can reject.
//!
//! Instrument, not a gate: asserts nothing. See the GAP-315 probe files.

#[allow(
    dead_code,
    reason = "every field is read by the `{:?}` in the printed report; derive(Debug) \
does not count as a read for dead_code, and the report IS this instrument's output"
)]
#[derive(Debug)]
enum Outcome {
    Ok2(usize, usize),
    Err(String),
    Panic(String),
}

fn probe(stride: &[isize]) -> Outcome {
    let owned: Vec<isize> = stride.to_vec();
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(move || fuel_ir::shape::stride_dims2(&owned));
    std::panic::set_hook(prev);
    match r {
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "<non-string panic>".into());
            Outcome::Panic(msg.lines().next().unwrap_or("").to_string())
        }
        Ok(Ok((a, b))) => Outcome::Ok2(a, b),
        Ok(Err(e)) => Outcome::Err(format!("{e}").lines().next().unwrap_or("").to_string()),
    }
}

#[test]
#[ignore = "instrument, not a gate: asserts nothing; see the module doc"]
fn gap315_stride_dims_predicate_probe() {
    println!(
        "\n=== GAP-315 predicate experiment (C): fuel_ir::shape::stride_dims2 ===\n\
         debug_assertions = {}  ({})",
        cfg!(debug_assertions),
        if cfg!(debug_assertions) {
            "ARM A - guard PRESENT (positive control)"
        } else {
            "ARM B - guard ABSENT (what the shipped artifact does)"
        }
    );

    // CONTROL: valid, non-negative. Must be identical in both arms.
    println!(
        "  [control] VALID   [4, 1]            -> {:?}",
        probe(&[4, 1])
    );

    // The LENGTH guard is a real `return Err` — it should reject in BOTH arms.
    // This is the in-probe control proving a release-surviving rejector exists
    // in this very function, so a silent Ok elsewhere is not "nothing can fail".
    println!(
        "  [len]     WRONG LEN [1, 2, 3]       -> {:?}",
        probe(&[1, 2, 3])
    );

    // 🔴 THE PREDICTION: negative stride. `as usize` wraps, nothing reconciles.
    println!(
        "  [neg]     NEGATIVE  [-1, 1]         -> {:?}",
        probe(&[-1, 1])
    );
    println!(
        "  [neg2]    NEGATIVE  [4, -8]         -> {:?}",
        probe(&[4, -8])
    );

    println!("  (measurement only - asserts nothing)");
}
