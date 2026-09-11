// SPDX-License-Identifier: MIT OR Apache-2.0
//! GAP-315 four-outcome probe, representative 3 — MEASUREMENT ONLY, not a gate.
//!
//! Sub-class: caller-supplied VALUE (list ordering and length), `pub` fn,
//! on a CROSS-PARTY SEAM — the advertisement this builds is consumed by the
//! other side of the kernel seam.
//!
//! ⚠️ `advertise`'s own doc comment says: "Never panics: a list longer than the
//! cap is truncated (a Fuel-side caller bug surfaced by the `debug_assert`s
//! rather than a release abort)." That sentence describes DEBUG. In release the
//! `debug_assert`s do not exist, so nothing surfaces the caller bug there.
//!
//! ⚠️ The (d) detector is NOTHING REJECTED IT — `advertise` returns AND
//! `validate` accepts. It is not a value comparison.
//!
//! Run twice:
//!   arm A  (default)                          debug_assertions ON   guard present
//!   arm B  RUSTFLAGS=-C debug-assertions=off  debug_assertions OFF  guard ABSENT
//!
//! ROW: GAP-317 reproducer
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
use fuel_kernel_seam_announce::{SEAM_MAX_PROFILES, SeamHello, negotiate};

#[allow(
    dead_code,
    reason = "every field here is read by the `{:?}` in the printed report; \n`derive(Debug)` does not count as a read for the dead_code lint, and the \nreport IS this instrument's entire output"
)]
#[derive(Debug)]
enum Outcome {
    Panic(String),
    /// advertise returned; `validate` then accepted or rejected it.
    Built {
        advertised_len: usize,
        validate: String,
    },
}

fn probe(profiles: Vec<u16>) -> Outcome {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(move || {
        let h = SeamHello::advertise(&profiles, 0);
        let v = match h.validate() {
            Ok(live) => format!("Ok(live_len={})", live.len()),
            Err(e) => format!("Err({e:?})"),
        };
        (h.profiles_len as usize, v)
    });
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
        Ok((n, v)) => Outcome::Built {
            advertised_len: n,
            validate: v,
        },
    }
}

#[test]
#[ignore = "instrument, not a gate: asserts nothing; see the module doc"]
fn gap315_seam_advertise_probe() {
    println!(
        "\n=== GAP-315 probe 3: SeamHello::advertise (pub fn, caller VALUE, cross-party seam) ===\n\
         debug_assertions = {}  ({})\n\
         SEAM_MAX_PROFILES = {SEAM_MAX_PROFILES}",
        cfg!(debug_assertions),
        if cfg!(debug_assertions) {
            "ARM A - guard PRESENT (positive control)"
        } else {
            "ARM B - guard ABSENT (what the shipped artifact does)"
        }
    );

    println!(
        "  [control] VALID ascending [1,2,3]            -> {:?}",
        probe(vec![1, 2, 3])
    );

    // (x) ordering violated. `validate` DOES check ordering and returns a typed
    //     Err, so the rejector survives release -- this should be (a1).
    println!(
        "  [x]   NOT ascending [3,1,2]                  -> {:?}",
        probe(vec![3, 1, 2])
    );

    // (xi) ⚠️ THE (d) CANDIDATE. Over-long list. `advertise` SILENTLY TRUNCATES
    //      to SEAM_MAX_PROFILES, and `validate` then sees a well-formed envelope
    //      with n <= MAX, so it ACCEPTS. Nothing anywhere reports that profiles
    //      the caller asked to advertise were dropped.
    let over: Vec<u16> = (1..=(SEAM_MAX_PROFILES as u16 + 4)).collect();
    println!(
        "  [xi]  {} profiles (cap {SEAM_MAX_PROFILES})                     -> {:?}",
        over.len(),
        probe(over.clone())
    );

    // (xii) THE OBSERVABLE HARM of (xi), at the seam. Two peers that genuinely
    //       share profile 20. The local side asks to advertise 1..=20; release
    //       truncates to 1..=16, so 20 is gone and nothing said so.
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(|| {
        let local = SeamHello::advertise(&(1..=20u16).collect::<Vec<_>>(), 0);
        let remote = SeamHello::advertise(&[20u16], 0);
        format!("{:?}", negotiate(&local, &remote).map(|n| n.profile))
    });
    std::panic::set_hook(prev);
    println!(
        "  [xii] negotiate: both support 20, local asks 1..=20 -> {}",
        match r {
            Ok(s) => s,
            Err(_) => "PANIC (the guard fired)".to_string(),
        }
    );
    println!("        (both peers DO support profile 20; anything other than Ok(20) is the harm)");
    println!("  (measurement only - this test asserts nothing about the verdict)");
}
