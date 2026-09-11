// SPDX-License-Identifier: MIT OR Apache-2.0
//! GAP-315 four-outcome probe — MEASUREMENT ONLY, not a gate.
//!
//! Asks what a RELEASE build does with an input that violates a
//! `debug_assert` on caller-supplied shape. Run twice, toggling the ONE axis
//! under test:
//!
//!   arm A   (default)                          debug_assertions ON   guard present
//!   arm B   RUSTFLAGS=-C debug-assertions=off  debug_assertions OFF  guard ABSENT
//!
//! Arm A is the positive control: it proves the probe REACHES the guard. Without
//! it, arm B's result is an absence with nothing showing the input was invalid.
//!
//! ⚠️ The (d) detector is NOTHING REJECTED IT — build Ok AND realize Ok — never a
//! shape comparison. A wrong-shape result that still builds and realizes is the
//! silent-wrong outcome regardless of what the shape happens to read.
//!
//! ROW: GAP-315 / GAP-316 sibling probe
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
use fuel_core::Device;
use fuel_core::lazy::Tensor;

/// What actually happened, classified by OUTCOME rather than by value.
#[allow(
    dead_code,
    reason = "every field here is read by the `{:?}` in the printed report; \n`derive(Debug)` does not count as a read for the dead_code lint, and the \nreport IS this instrument's entire output"
)]
#[derive(Debug)]
enum Outcome {
    /// Typed `Err` from the build — a guard that survives release.
    BuildErr(String),
    /// Panicked — a guard that survives release, but as a panic.
    Panic(String),
    /// Built AND realized with nothing objecting. This is the (d) shape.
    OkRealized { dims: Vec<usize>, elems: usize },
    /// Built, but realize objected — outcome (b).
    OkBuildRealizeErr { dims: Vec<usize> },
}

fn cpu_f32(data: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape.to_vec(), &Device::cpu()).unwrap()
}

/// Run one probe, catching a panic so the classification is total.
fn probe<F>(f: F) -> Outcome
where
    F: FnOnce() -> std::result::Result<Tensor, fuel_ir::Error> + std::panic::UnwindSafe,
{
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // silence the backtrace spam
    let r = std::panic::catch_unwind(f);
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
        Ok(Err(e)) => Outcome::BuildErr(format!("{e}").lines().next().unwrap_or("").to_string()),
        Ok(Ok(t)) => {
            let dims = t.shape().dims().to_vec();
            let realized =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| t.realize_f32()));
            match realized {
                Ok(v) => Outcome::OkRealized {
                    dims,
                    elems: v.len(),
                },
                Err(_) => Outcome::OkBuildRealizeErr { dims },
            }
        }
    }
}

#[test]
#[ignore = "instrument, not a gate: asserts nothing; see the module doc"]
fn gap315_split_heads_probe() {
    let guard_present = cfg!(debug_assertions);
    println!(
        "\n=== GAP-315 probe: Tensor::split_heads (pub fn, caller SHAPE, lazy) ===\n\
         debug_assertions = {guard_present}  ({})",
        if guard_present {
            "ARM A - guard PRESENT (positive control: the probe must reach it)"
        } else {
            "ARM B - guard ABSENT (this is what the shipped artifact does)"
        }
    );

    // CONTROL: a VALID call. Must behave identically in both arms; if this
    // differs, the probe is measuring something other than the guard.
    let valid =
        probe(|| cpu_f32((0..12).map(|i| i as f32).collect(), &[1, 2, 6]).split_heads(2, 3));
    println!("  [control] VALID  [1,2,6].split_heads(2,3)      -> {valid:?}");

    // (i) dim violated, element COUNT also differs (the ordinary case).
    //     6 != 2*4, and 1*2*6=12 != 1*2*2*4=16, so a count-checking
    //     downstream can see it.
    let i = probe(|| cpu_f32((0..12).map(|x| x as f32).collect(), &[1, 2, 6]).split_heads(2, 4));
    println!("  [i]   dim violated, COUNT DIFFERS             -> {i:?}");

    // (ii) ⚠️ COUNT-PRESERVING violation. b*n == 0 makes both totals 0, so a
    //      downstream that validates ELEMENT COUNT cannot see a violated DIM.
    //      This is the (a1-PARTIAL) divergence input for this site.
    let ii = probe(|| cpu_f32(vec![], &[0, 2, 6]).split_heads(2, 4));
    println!("  [ii]  dim violated, COUNT PRESERVED (b=0)     -> {ii:?}");

    // ⚠️ ZERO-SIZE CONTROLS FOR [ii]. The [ii] fixture uses b=0 to make the two
    //    element counts collide — but that ALSO makes the tensor empty, and an
    //    empty tensor may fail to realize for reasons having nothing to do with
    //    the violated dim. Without these, a realize failure in [ii] cannot be
    //    attributed: the fixture would have collapsed the axis it was built to test.
    //    z1 is a VALID split on the SAME zero-size tensor (6 == 2*3).
    //    z2 realizes the zero-size tensor with no split_heads at all.
    let z1 = probe(|| cpu_f32(vec![], &[0, 2, 6]).split_heads(2, 3));
    println!("  [z1]  CONTROL zero-size, VALID split(2,3)     -> {z1:?}");
    let z2 = probe(|| Ok(cpu_f32(vec![], &[0, 2, 6])));
    println!("  [z2]  CONTROL zero-size, NO split_heads       -> {z2:?}");

    // (iii) rank violated: rank 1. Both guards gone, dims[1] is indexed directly.
    let iii = probe(|| cpu_f32(vec![0.0; 6], &[6]).split_heads(2, 3));
    println!("  [iii] rank violated (rank 1)                  -> {iii:?}");

    // (iv) rank violated: rank 2, count differs.
    let iv = probe(|| cpu_f32(vec![0.0; 12], &[2, 6]).split_heads(2, 3));
    println!("  [iv]  rank violated (rank 2)                  -> {iv:?}");

    println!("  (measurement only - this test asserts nothing about the verdict)");
}
