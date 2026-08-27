// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared test-gate helpers — **a missing prerequisite is a failure, not a skip.**
//!
//! # The defect this crate exists to remove (GAP-157)
//!
//! **An early `return` from a `#[test]` is a `PASS`.** A test that asks for a
//! device, does not get one, and returns reports `ok` having asserted nothing —
//! and *no coverage mechanism can see it*, because the test ran, exited
//! normally, and was counted as passing. The registry's control-flow
//! enumeration found **502 such tests across the repo, 86% of them silent**
//! (they print nothing on the skip path, so a `grep` for `SKIP` finds fewer than
//! one in seven).
//!
//! The idiom that produces them:
//!
//! ```ignore
//! fn dev_or_skip() -> Option<CudaDevice> { CudaDevice::new(0).ok() }
//!
//! #[ignore]
//! #[test]
//! fn matmul_matches_reference() {
//!     let Some(dev) = dev_or_skip() else { return };  // <-- reports `ok`
//!     assert_eq!(..);
//! }
//! ```
//!
//! That test is `#[ignore]`d, so it runs **only** when a human explicitly passes
//! `--ignored`. The skip is therefore not merely silent, it is *incoherent*:
//! **you asked for this test by name and it decided not to run.** The `#[ignore]`
//! is already the declared gate; the runtime skip inside it is a second,
//! undeclared one.
//!
//! # What to use instead
//!
//! ```ignore
//! use fuel_test_support::required;
//!
//! fn dev_or_fail() -> CudaDevice {
//!     required("a live CUDA device", CudaDevice::new(0).ok())
//! }
//! ```
//!
//! # A skip that is CORRECT is not a defect — but it must be DECLARED
//!
//! This crate does not claim every skip is wrong. A test that genuinely cannot
//! run in an environment *should* not run there. The rule is that the gate must
//! be **declarable and discoverable** — `#[ignore]`, or `#[cfg(feature = "…")]`
//! — so a reader (and a harness) can see it without executing the test. What
//! this crate removes is the *undeclared* gate: the one that exists only as
//! control flow inside a body, invisible to every mechanism except reading it.
//!
//! # Why this crate carries no `cfg` and no features
//!
//! These helpers take `Option<T>` / `Result<T, E>` and hand back `T`. They name
//! no device, no backend and no dtype, so they need **no** dependency on
//! `fuel-cuda-backend`, `fuel-vulkan-backend` or anything else — and therefore
//! have **nothing to gate**.
//!
//! That is deliberate, and it is the design constraint the registry raised
//! before this crate was built: *the shared helper must be correctly gated
//! across cuda-only / vulkan-only / both, and MUST NOT be able to compile out
//! into nothing.* A single point of failure whose miswiring would be invisible
//! is precisely the shape of the bug being fixed. **The constraint is dissolved
//! rather than satisfied: a crate with no `cfg` at all cannot compile out**, so
//! there is no gating to get wrong.

pub mod gpu_run_lock;
pub mod hardware;

pub use gpu_run_lock::{GpuRunLock, UNGUARDED_VAR, require_gpu_run_lock};

use std::fmt::Display;

/// The remedy sentence appended to every failure this crate raises.
///
/// Kept in one place because it is the part that tells the next reader what to
/// do *instead* — without it the panic says only "no device", which reads as a
/// broken machine rather than as a missing declaration.
const REMEDY: &str = "a missing prerequisite is a FAILURE, not a skip (GAP-157): this test ran and \
     would otherwise have reported `ok` having asserted nothing. If it genuinely \
     cannot run in this environment, DECLARE that with `#[ignore]` or \
     `#[cfg(feature = \"…\")]` so the gate is discoverable, instead of returning early";

/// Unwrap a prerequisite that the test **requires**, panicking with an
/// explanation if it is absent.
///
/// `what` names the missing thing from the test's point of view — *"a live CUDA
/// device"*, *"an AMD integrated GPU adapter"* — and is quoted back in the
/// failure, so the reader learns which prerequisite was unmet without opening
/// the file.
///
/// `#[track_caller]` makes the panic point at the *call site* in the test rather
/// than at this crate, so the failure names the test that needs the device.
#[track_caller]
pub fn required<T>(what: &str, got: Option<T>) -> T {
    match got {
        Some(v) => v,
        None => panic!("{what} is REQUIRED by this test and was not available — {REMEDY}"),
    }
}

/// As [`required`], for a prerequisite that reports *why* it was unavailable.
///
/// Prefer this over `.ok()` + [`required`] whenever the source is a `Result`:
/// the error is the difference between *"no CUDA device"* and *"CUDA device
/// present but the driver refused: out of memory"*, and discarding it turns a
/// diagnosable failure into a mysterious one.
#[track_caller]
pub fn required_ok<T, E: Display>(what: &str, got: Result<T, E>) -> T {
    match got {
        Ok(v) => v,
        Err(e) => panic!("{what} is REQUIRED by this test and was not available ({e}) — {REMEDY}"),
    }
}

/// As [`required`], for a prerequisite already reduced to a `bool`.
///
/// Exists because the common Fuel guard is a predicate — `cuda_present()`,
/// `vulkan_present()` — rather than an `Option`. Writing that as
/// `required(what, present.then_some(()))` works but reads as a puzzle at a
/// call site, and a helper nobody can read at a glance is one people route
/// around.
///
/// Carries strictly less information than [`required_ok`]: a `bool` has already
/// discarded *why*. Prefer the `Result` form when the source has one.
#[track_caller]
pub fn require(what: &str, present: bool) {
    if !present {
        panic!("{what} is REQUIRED by this test and was not available — {REMEDY}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Catch a panic and hand back its message as a `String`.
    fn panic_message<F: FnOnce() + std::panic::UnwindSafe>(f: F) -> String {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let payload = std::panic::catch_unwind(f).expect_err("expected a panic, got none");
        std::panic::set_hook(prev);
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .expect("panic payload was neither String nor &str")
    }

    #[test]
    fn required_passes_the_value_through_when_present() {
        assert_eq!(required("a device", Some(7u32)), 7);
        assert_eq!(required_ok::<u32, String>("a device", Ok(7)), 7);
    }

    /// The whole point of the crate: absence must be LOUD.
    #[test]
    fn required_panics_when_absent() {
        panic_message(|| {
            required::<u32>("a live CUDA device", None);
        });
        panic_message(|| {
            required_ok::<u32, String>("a live CUDA device", Err("driver refused".into()));
        });
    }

    /// Panicking is necessary but NOT sufficient. A bare `.unwrap()` also
    /// panics, and its message (`called Option::unwrap() on a None value`)
    /// tells the reader nothing about which prerequisite was missing or what to
    /// do about it — it reads as a broken machine, not a missing declaration.
    /// These assertions are what separate this helper from `.unwrap()`.
    #[test]
    fn the_failure_explains_itself() {
        let msg = panic_message(|| {
            required::<u32>("a live CUDA device", None);
        });
        assert!(
            msg.contains("a live CUDA device"),
            "the failure must NAME the missing prerequisite; got: {msg}"
        );
        assert!(
            msg.contains("not a skip"),
            "the failure must say a missing prerequisite is a failure, not a skip; got: {msg}"
        );
        assert!(
            msg.contains("#[ignore]") && msg.contains("cfg(feature"),
            "the failure must point at the DECLARED-gate remedy, or the reader's \
             cheapest fix is to put the silent early return back; got: {msg}"
        );
    }

    #[test]
    fn require_is_a_no_op_when_present_and_explains_when_not() {
        require("a live CUDA device", true); // must not panic
        let msg = panic_message(|| require("a live CUDA device", false));
        assert!(
            msg.contains("a live CUDA device"),
            "must NAME it; got: {msg}"
        );
        assert!(
            msg.contains("not a skip"),
            "must say failure-not-skip; got: {msg}"
        );
        assert!(
            msg.contains("#[ignore]") && msg.contains("cfg(feature"),
            "must point at the declared-gate remedy; got: {msg}"
        );
    }

    /// The error cause must survive. Dropping it turns "the driver refused: out
    /// of memory" into "no device", which is a different and much harder bug.
    #[test]
    fn required_ok_keeps_the_underlying_error() {
        let msg = panic_message(|| {
            required_ok::<u32, String>("a live CUDA device", Err("driver refused: OOM".into()));
        });
        assert!(
            msg.contains("driver refused: OOM"),
            "the underlying error must be preserved; got: {msg}"
        );
    }
}
