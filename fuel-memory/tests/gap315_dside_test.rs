// SPDX-License-Identifier: MIT OR Apache-2.0
//! GAP-315 — the genuine (d)-SIDE FORWARD TEST. MEASUREMENT ONLY, not a gate.
//!
//! ⚠️ PREDICTIONS AND THE CORRECT ANSWER WERE REGISTERED BEFORE THIS RAN.
//!
//! SITE: `Storage::with_bundle`. Chosen because it satisfies every clause of the
//! spec the earlier probes could not:
//!   - returns `Result<Self>`, and the error path is LIVE (not theoretical):
//!     `validate_bundle` rejects an empty bundle and a slot-0 dtype mismatch.
//!     Arm [ctl-err] below exercises that, so a silent `Ok` elsewhere cannot be
//!     explained by "nothing in this function can fail".
//!   - nothing on that error path checks the asserted quantity: `validate_bundle`
//!     validates the NEW bundle against the dtype and never looks at the existing one.
//!   - no second mechanism forces the outcome — no `as usize` wrap, no missing
//!     error variant, no bounds check in play.
//!
//! 🔴 THE CORRECT ANSWER, NAMED FROM THE CONTRACT BEFORE PROBING:
//!   the function's own doc says "panics in debug mode if a bundle is already
//!   attached (RE-BUNDLING IS A CONTRACT BUG)". So for a re-bundle the callee owes
//!   a REJECTION — or at minimum must not silently discard the first bundle.
//!   PREDICTED release behaviour: `Ok` with the SECOND bundle installed and the
//!   FIRST silently gone  => (d).
//!
//! ⚠️ WHY THIS IS NOT THE conv CASE: the violation and the harm are THE SAME ACT.
//! Re-bundling is the contract bug and its immediate consequence is that the
//! caller's first bundle is destroyed. No second caller error is needed to reach
//! the harm — which is the test conv failed and `advertise` passed.
//!
//! Run both arms per the other GAP-315 probes:
//!   arm A  (default)                          debug_assertions ON   guard present
//!   arm B  RUSTFLAGS="-C target-cpu=native -C debug-assertions=off"  guard ABSENT

use fuel_cpu_backend::byte_storage::CpuStorageBytes;
use fuel_ir::storage::{OutputViewSpec, compose_bundle};
use fuel_ir::{DType, Shape};
use fuel_memory::{BackendStorage, Storage};
use std::sync::Arc;

#[allow(
    dead_code,
    reason = "every field is read by the `{:?}` in the printed report; derive(Debug) \
does not count as a read for dead_code, and the report IS this instrument's output"
)]
#[derive(Debug)]
enum Outcome {
    /// Rejected with a typed error — a guard that survives release.
    Err(String),
    Panic(String),
    /// Accepted. `slots` is how many views the resulting Storage carries, which
    /// is how we tell WHICH bundle survived. This is the (d) shape.
    Ok {
        slots: usize,
    },
}

fn specs_of(n: usize) -> Vec<OutputViewSpec> {
    (0..n)
        .map(|i| OutputViewSpec::contiguous(DType::F32, Shape::from_dims(&[4 + i])))
        .collect()
}

/// Build a Storage carrying a bundle of `n` slots, all F32.
fn bundled(n: usize) -> Storage {
    let (total_bytes, views) = compose_bundle(&specs_of(n)).expect("compose_bundle");
    Storage::new_bundled(
        BackendStorage::Cpu(CpuStorageBytes::from_zero_bytes(total_bytes)),
        DType::F32,
        Arc::from(views.into_boxed_slice()),
    )
    .expect("new_bundled")
}

/// A bundle of `n` F32 slots, to hand to `with_bundle`.
fn bundle_of(n: usize) -> Arc<[fuel_ir::storage::OutputView]> {
    let (_bytes, views) = compose_bundle(&specs_of(n)).expect("compose_bundle");
    Arc::from(views.into_boxed_slice())
}

fn probe<F: FnOnce() -> fuel_ir::Result<Storage> + std::panic::UnwindSafe>(f: F) -> Outcome {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(f);
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
        Ok(Err(e)) => Outcome::Err(format!("{e}").lines().next().unwrap_or("").to_string()),
        Ok(Ok(s)) => Outcome::Ok {
            slots: s.bundle().map(|b| b.len()).unwrap_or(0),
        },
    }
}

#[test]
#[ignore = "instrument, not a gate: asserts nothing; see the module doc"]
fn gap315_with_bundle_dside_probe() {
    println!(
        "\n=== GAP-315 (d)-SIDE FORWARD TEST: Storage::with_bundle ===\n\
         debug_assertions = {}  ({})",
        cfg!(debug_assertions),
        if cfg!(debug_assertions) {
            "ARM A - guard PRESENT (positive control)"
        } else {
            "ARM B - guard ABSENT (what the shipped artifact does)"
        }
    );

    // CONTROL: the legitimate use — attach a bundle to an UNBUNDLED Storage.
    // Must behave identically in both arms.
    println!(
        "  [control]  unbundled + with_bundle(2)   -> {:?}",
        probe(|| {
            let s = Storage::new(
                BackendStorage::Cpu(CpuStorageBytes::from_zero_bytes(64)),
                DType::F32,
            );
            s.with_bundle(bundle_of(2))
        })
    );

    // IN-PROBE CONTROL: the error path is LIVE. An empty bundle must be REJECTED
    // in BOTH arms. Without this, a silent Ok below could be explained away as
    // "nothing in this function can fail".
    println!(
        "  [ctl-err]  unbundled + EMPTY bundle     -> {:?}",
        probe(|| {
            let s = Storage::new(
                BackendStorage::Cpu(CpuStorageBytes::from_zero_bytes(64)),
                DType::F32,
            );
            s.with_bundle(Arc::from(Vec::new().into_boxed_slice()))
        })
    );

    // 🔴 THE PREDICTION: RE-BUNDLE. Storage already carries 3 slots; attach 2 more.
    //    Contract says re-bundling is a bug, so the callee owes a rejection.
    //    PREDICT arm B: Ok { slots: 2 } — the FIRST bundle (3 slots) silently gone.
    println!(
        "  [rebundle] bundled(3) + with_bundle(2)  -> {:?}",
        probe(|| bundled(3).with_bundle(bundle_of(2)))
    );

    // Same violation, sizes swapped, so "slots" cannot be read as a coincidence.
    println!(
        "  [rebundle2] bundled(2) + with_bundle(5) -> {:?}",
        probe(|| bundled(2).with_bundle(bundle_of(5)))
    );

    println!("  (measurement only - asserts nothing)");
}
