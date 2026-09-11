// SPDX-License-Identifier: MIT OR Apache-2.0
//! GAP-315 four-outcome probe, representative 2 — MEASUREMENT ONLY, not a gate.
//!
//! `conv2d_direct` is a `pub` fn whose caller supplies FOUR buffers, each guarded
//! by a `debug_assert_eq!` on its length against a geometry computed from
//! `ConvShape`. Those guards are absent from the shipped artifact.
//!
//! Its own doc states the contract the guards enforce:
//!   "`out` must be sized `s.output_len()`. The function writes every output
//!    element exactly once; pre-zeroing isn't required."
//!
//! ⚠️ The (d) detector is NOTHING REJECTED IT — the call returns normally on a
//! known-invalid input. It is NOT a value comparison. The value is reported
//! afterwards only to characterise severity, never to decide the outcome.
//!
//! Run twice:
//!   arm A  (default)                          debug_assertions ON   guard present
//!   arm B  RUSTFLAGS=-C debug-assertions=off  debug_assertions OFF  guard ABSENT
//!
//! ROW: GAP-316 reproducer
//!
//! ══════════════════════════════════════════════════════════════════════════
//! ⚠️ THIS IS AN INSTRUMENT, NOT A GATE. IT IS `#[ignore]`d ON PURPOSE.
//! ══════════════════════════════════════════════════════════════════════════
//!
//! It ASSERTS NOTHING about correctness. It prints what the code does and stops.
//! Left un-ignored it would report `ok` on every run while checking nothing —
//! a vacuous green that reads as coverage. So it is `#[ignore]`d, which means it
//! still COMPILES on every CI run (so it cannot rot unnoticed) but never RUNS
//! (so it can never masquerade as a passing test).
//!
//! ⚠️ `#[ignore]` on its own reads as "flaky" or "slow". It is neither. The
//! reason is this paragraph, and it must survive any edit to the attribute.
//!
//! WHY IT IS KEPT: it is the reproducer for a MEASURED defect. Deleting it makes
//! the measurement unrepeatable by anyone who was not here.
//!
//! ⚠️ WHAT IT BECOMES WHEN THE DEFECT IS FIXED — this is the whole reason a
//! PRESERVED reproducer beats a REBUILT one:
//!   the fixer UN-IGNORES it and INVERTS its assertion, and it is then the
//!   BORN-RED that proves the fix. It was written BEFORE anyone knew what the
//!   fix would be, so it CANNOT have been shaped to fit that fix. A probe
//!   reconstructed afterwards always can be, and nothing in it would show the
//!   difference.
//!
//! HOW TO RUN IT:
//!   arm A (guard present):  cargo test -p <crate> --test <this> -- --ignored --nocapture
//!   arm B (guard ABSENT, = the shipped configuration):
//!     RUSTFLAGS="-C target-cpu=native -C debug-assertions=off" \
//!     CARGO_TARGET_DIR=<a separate dir> \
//!     cargo test -p <crate> --test <this> -- --ignored --nocapture
//!
//! ⚠️ `-C target-cpu=native` is NOT decoration: an env `RUSTFLAGS` REPLACES
//! `.cargo/config.toml`'s `[build] rustflags` rather than appending to it, so
//! dropping it varies the two arms on TWO axes and no difference can then be
//! attributed to the guard. Use a separate `CARGO_TARGET_DIR` so the two
//! configurations do not invalidate each other's cache on every switch.
use fuel_conv::{ConvShape, conv2d_direct};

#[allow(
    dead_code,
    reason = "every field here is read by the `{:?}` in the printed report; \n`derive(Debug)` does not count as a read for the dead_code lint, and the \nreport IS this instrument's entire output"
)]
#[derive(Debug)]
enum Outcome {
    /// Nothing objected. On a known-invalid input this is the (d) shape.
    Returned {
        wrote: usize,
        tail_untouched: usize,
    },
    Panic(String),
}

fn shape() -> ConvShape {
    ConvShape {
        batch: 1,
        c_in: 1,
        c_out: 1,
        h: 3,
        w: 3,
        k_h: 2,
        k_w: 2,
        stride: (1, 1),
        padding: (0, 0),
        groups: 1,
    }
}

/// `SENTINEL` must not be a value the convolution could legitimately produce,
/// or "the tail was not written" and "the tail was written with this value"
/// become indistinguishable.
const SENTINEL: f32 = -99999.0;

fn run(x_len: usize, w_len: usize, out_len: usize) -> Outcome {
    let s = shape();
    let x: Vec<f32> = (0..x_len).map(|i| (i + 1) as f32).collect();
    let w: Vec<f32> = (0..w_len).map(|_| 1.0).collect();
    let mut out = vec![SENTINEL; out_len];

    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        conv2d_direct(&x, &w, None, &s, &mut out);
    }));
    std::panic::set_hook(prev);

    match r {
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "<non-string panic>".to_string());
            Outcome::Panic(msg.lines().next().unwrap_or("").to_string())
        }
        Ok(()) => Outcome::Returned {
            wrote: out.iter().filter(|v| **v != SENTINEL).count(),
            tail_untouched: out.iter().filter(|v| **v == SENTINEL).count(),
        },
    }
}

#[test]
#[ignore = "instrument, not a gate: asserts nothing; see the module doc"]
fn gap315_conv2d_direct_probe() {
    let s = shape();
    let valid_x = s.batch * s.c_in * s.h * s.w; // 9
    let valid_w = s.c_out * s.c_in * s.k_h * s.k_w; // 4
    let valid_out = s.output_len(); // 1*1*2*2 = 4

    println!(
        "\n=== GAP-315 probe 2: conv2d_direct (pub fn, caller SHAPE, eager buffers) ===\n\
         debug_assertions = {}  ({})\n\
         valid sizes: x={valid_x} weight={valid_w} out={valid_out}",
        cfg!(debug_assertions),
        if cfg!(debug_assertions) {
            "ARM A - guard PRESENT (positive control)"
        } else {
            "ARM B - guard ABSENT (what the shipped artifact does)"
        }
    );

    println!(
        "  [control] all VALID                          -> {:?}",
        run(valid_x, valid_w, valid_out)
    );
    // ⚠️ THE (d) CANDIDATE. An OVER-LONG `out` violates the assert, and the
    //    function's own contract says it writes every output element exactly
    //    once -- so the tail is left as the caller found it, with nothing
    //    reporting that the buffer was the wrong size.
    println!(
        "  [v]   out TOO LONG  ({} vs {valid_out})            -> {:?}",
        valid_out + 4,
        run(valid_x, valid_w, valid_out + 4)
    );
    println!(
        "  [vi]  out TOO SHORT ({} vs {valid_out})            -> {:?}",
        valid_out - 1,
        run(valid_x, valid_w, valid_out - 1)
    );
    println!(
        "  [vii] x TOO SHORT   ({} vs {valid_x})            -> {:?}",
        valid_x - 1,
        run(valid_x - 1, valid_w, valid_out)
    );
    println!(
        "  [viii] x TOO LONG   ({} vs {valid_x})            -> {:?}",
        valid_x + 5,
        run(valid_x + 5, valid_w, valid_out)
    );
    println!(
        "  [ix]  weight TOO SHORT ({} vs {valid_w})         -> {:?}",
        valid_w - 1,
        run(valid_x, valid_w - 1, valid_out)
    );
    println!("  (measurement only - this test asserts nothing about the verdict)");
}
