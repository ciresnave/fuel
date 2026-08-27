// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Hardware-gated skips, and the policy that decides whether a skip is fatal.**
//!
//! # What this adds to the crate, and why the existing helpers were not enough
//!
//! [`crate::required`] / [`crate::required_ok`] make a missing prerequisite an
//! *unconditional* failure. That is right when the test cannot mean anything
//! without the device, and it is what most of this crate is for.
//!
//! It is wrong for a test that is legitimately run on machines that differ.
//! `cargo test -- --ignored` on a box with CUDA but no AOCL runs *every* ignored
//! test; if every one of them failed hard, the run would be red for families of
//! hardware nobody claimed to have, and the first thing anyone would do is stop
//! running it that way. **A gate people switch off protects nothing.**
//!
//! So this module adds the missing middle: absence is a *declared skip* by
//! default, and becomes a *failure* in an environment that promised the
//! hardware. That is the shape vulkane arrived at (`VULKANE_REQUIRE_DEVICE`),
//! and the reasoning below is theirs, adapted.
//!
//! # Why one variable per family, and not one `FUEL_REQUIRE_DEVICE`
//!
//! vulkane has a single variable because it has a single device class. Fuel has
//! four, and **they are independently present**: measured on the development box
//! 2026-08-27, CUDA and Vulkan are available while AOCL and MKL are not (their
//! DLLs are absent -- see GAP-016, and both crates are excluded from CI for the
//! same reason).
//!
//! A single variable therefore cannot express the truth about any real machine
//! here. It would force a choice between "CUDA skips are silent" and "AOCL skips
//! are fatal on a box that has never had AOCL", and both are wrong. One variable
//! per family lets a machine declare exactly what it has.
//!
//! # The load-bearing split: [`Missing::Device`] vs [`Missing::Capability`]
//!
//! Carried over from vulkane, whose asymmetry argument is the reason it exists:
//!
//! - A **capability** gate misrouted to [`Missing::Device`] turns a run red for
//!   hardware that is behaving correctly. Loud, wrong, and someone will "fix" it
//!   by unsetting the variable -- taking the whole mechanism with it.
//! - **Device absence** misrouted to [`Missing::Capability`] is *silent*. The run
//!   stays green and the evidence quietly stops being produced, which is the
//!   exact defect (GAP-157, GAP-243) this module exists to end.
//!
//! The two failure directions are not symmetric, so the classification is made
//! once, where the cause is known, and [`skip`] routes it mechanically.
//!
//! # Known limit, stated because it bounds what this buys
//!
//! `cargo test` captures output and echoes it only for a *failing* test, so the
//! `SKIP` line below is invisible in a green run without `--nocapture`.
//! **Declaring the skip does not by itself make it visible in CI.** The thing
//! that makes a skip impossible to miss is the variable turning it into a
//! failure; the printed line is for a human reading a local run.

use std::ffi::OsStr;

/// A class of accelerator Fuel tests gate on.
///
/// `Copy` and fieldless: a call site names one of these and nothing else, so a
/// site cannot invent a family that has no variable behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Hardware {
    Cuda,
    Vulkan,
    Aocl,
    Mkl,
}

impl Hardware {
    /// Every family. Exists so tests can range over the set rather than
    /// re-listing it -- a hand-written list in a test is the thing that silently
    /// stops covering a variant added later.
    pub const ALL: [Hardware; 4] = [
        Hardware::Cuda,
        Hardware::Vulkan,
        Hardware::Aocl,
        Hardware::Mkl,
    ];

    /// Short lower-case tag used in the printed `SKIP` line.
    pub const fn tag(self) -> &'static str {
        match self {
            Hardware::Cuda => "cuda",
            Hardware::Vulkan => "vulkan",
            Hardware::Aocl => "aocl",
            Hardware::Mkl => "mkl",
        }
    }

    /// The environment variable that makes this family's device-absence fatal.
    ///
    /// **These must be distinct across families** -- a collapsed mapping would
    /// make one family's variable silently arm another's, and the failure would
    /// be a red run blamed on the wrong hardware. Asserted in tests.
    pub const fn require_var(self) -> &'static str {
        match self {
            Hardware::Cuda => "FUEL_REQUIRE_CUDA",
            Hardware::Vulkan => "FUEL_REQUIRE_VULKAN",
            Hardware::Aocl => "FUEL_REQUIRE_AOCL",
            Hardware::Mkl => "FUEL_REQUIRE_MKL",
        }
    }
}

/// What a test needed and did not get -- carrying **which class of absence it
/// is**, not merely a message.
///
/// See the module docs for why the class travels with the cause instead of
/// being re-decided at each call site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Missing {
    /// No usable device of this family. An environment that promised one is
    /// misconfigured, so this is fatal under the family's variable.
    Device(String),
    /// A device is present but does not offer what the test needs. Conformant
    /// hardware, therefore **never** fatal.
    Capability(String),
}

impl Missing {
    /// No driver, no device, no adapter -- the family is simply not here.
    pub fn device(reason: impl Into<String>) -> Self {
        Missing::Device(reason.into())
    }

    /// An optional extension, feature, dtype, or compute capability the present
    /// device does not support.
    pub fn capability(what: impl Into<String>) -> Self {
        Missing::Capability(what.into())
    }
}

/// Decide fatality from a variable's raw value, without touching the process
/// environment.
///
/// Split out from [`skips_are_fatal`] so the reading rule is testable directly.
/// `std::env::set_var` is `unsafe` in edition 2024 and racy under the parallel
/// test harness, so a test that mutated the real environment would be both
/// unsound and flaky -- this function lets every branch be covered with neither.
///
/// Absent or empty is **not** set: `FUEL_REQUIRE_CUDA=` from a shell that always
/// exports its variables must not arm the gate.
pub fn value_arms_the_gate(value: Option<&OsStr>) -> bool {
    value.is_some_and(|v| !v.is_empty())
}

/// Whether device-absence for `hw` is currently fatal.
pub fn skips_are_fatal(hw: Hardware) -> bool {
    value_arms_the_gate(std::env::var_os(hw.require_var()).as_deref())
}

/// Declare a [`Missing`], routing to the right behaviour by its class.
///
/// Returns `()`, so a call site declares and bails in one statement:
///
/// ```ignore
/// let dev = match CudaDevice::new(0) {
///     Ok(d) => d,
///     Err(e) => return skip(Hardware::Cuda, Missing::device(format!("{e:?}"))),
/// };
/// ```
#[track_caller]
pub fn skip(hw: Hardware, cause: Missing) {
    skip_with(hw, cause, skips_are_fatal(hw));
}

/// [`skip`] with the policy decision supplied rather than read from the
/// environment. The whole of the behaviour lives here so both branches are
/// reachable from a test without setting a variable.
#[track_caller]
pub fn skip_with(hw: Hardware, cause: Missing, fatal: bool) {
    match cause {
        // Conformant hardware. Never fatal, whatever the variable says.
        Missing::Capability(what) => {
            eprintln!(
                "SKIP[{}]: device present but lacks {what} -- capability gates are never fatal",
                hw.tag()
            );
        }
        Missing::Device(reason) => {
            if fatal {
                panic!(
                    "no {} device, and {} is set — {reason}.\n\n\
                     This environment DECLARED it has {} hardware, so a skip here is not \
                     \"no device on this machine\", it is \"the evidence this run was \
                     supposed to produce did not get produced\". Either fix the \
                     environment, or unset {} if the promise was wrong.",
                    hw.tag(),
                    hw.require_var(),
                    hw.tag(),
                    hw.require_var()
                );
            }
            eprintln!("SKIP[{}]: {reason}", hw.tag());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panic_message<F: FnOnce() + std::panic::UnwindSafe>(f: F) -> String {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let payload = std::panic::catch_unwind(f).expect_err("expected a panic, got none");
        std::panic::set_hook(prev);
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .expect("panic payload was neither String nor &str")
    }

    /// THE point of the module: armed, a missing device is a FAILURE.
    #[test]
    fn an_armed_gate_turns_device_absence_into_a_failure() {
        let msg = panic_message(|| {
            skip_with(
                Hardware::Cuda,
                Missing::device("CudaDevice::new(0) returned NoDevice"),
                true,
            )
        });
        assert!(
            msg.contains("FUEL_REQUIRE_CUDA"),
            "the failure must name the variable that armed it, so a reader can \
             tell an unmet promise from a broken machine. Got: {msg}"
        );
        assert!(
            msg.contains("CudaDevice::new(0) returned NoDevice"),
            "the failure must quote the cause. Got: {msg}"
        );
    }

    /// The other branch, and the reason the mechanism is adoptable at all.
    #[test]
    fn an_unarmed_gate_returns_so_a_mixed_hardware_run_is_not_red() {
        skip_with(Hardware::Vulkan, Missing::device("no Vulkan ICD"), false);
    }

    /// A capability gate is never fatal -- even armed. This is the arm whose
    /// absence would make people switch the mechanism off.
    #[test]
    fn a_capability_gate_is_never_fatal_even_when_armed() {
        skip_with(
            Hardware::Cuda,
            Missing::capability("bf16 tensor cores"),
            true,
        );
    }

    /// A collapsed family -> variable mapping would arm the wrong hardware.
    #[test]
    fn every_family_has_a_distinct_require_var() {
        let mut seen = std::collections::BTreeSet::new();
        for hw in Hardware::ALL {
            assert!(
                seen.insert(hw.require_var()),
                "{:?} shares its variable {} with an earlier family -- one \
                 family's variable would silently arm another's, and the red \
                 run would name the wrong hardware",
                hw,
                hw.require_var()
            );
            assert!(
                hw.require_var().starts_with("FUEL_REQUIRE_"),
                "{:?} -> {}",
                hw,
                hw.require_var()
            );
        }
        assert_eq!(seen.len(), Hardware::ALL.len());
    }

    /// Absent and empty must both leave the gate unarmed.
    #[test]
    fn only_a_non_empty_value_arms_the_gate() {
        assert!(!value_arms_the_gate(None), "absent must not arm");
        assert!(
            !value_arms_the_gate(Some(OsStr::new(""))),
            "empty must not arm: a shell that exports every variable would \
             otherwise arm every family it names"
        );
        assert!(value_arms_the_gate(Some(OsStr::new("1"))));
        assert!(
            value_arms_the_gate(Some(OsStr::new("0"))),
            "non-empty is set"
        );
    }
}
