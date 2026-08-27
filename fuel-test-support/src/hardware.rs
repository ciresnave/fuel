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

    /// What absence means for this family when nothing overrides it.
    ///
    /// # ⚠️ This keys on a PROXY, and the proxy's assumption is enforced elsewhere
    ///
    /// The tempting justification is *"building `--features cuda` proves you
    /// have CUDA hardware"*. **That is false and must not be written here.**
    /// Measured 2026-08-27: `baracuda-kernels-sys/build.rs:604` expects `nvcc`
    /// (and `:198` skips it under `DOCS_RS=1` -- an escape hatch exists because
    /// the requirement is otherwise real), so `--features cuda` requires the
    /// **SDK**. An SDK is a *compiler*, not a *device*: a box with CUDA
    /// installed and no GPU is an ordinary configuration, and is exactly what
    /// the `DOCS_RS` path serves.
    ///
    /// What actually justifies `Fatal` is that **every CUDA and Vulkan call
    /// site is inside an `#[ignore]`d test**, so running one requires an
    /// explicit `-- --ignored`, which IS a declaration that the device is
    /// expected. `#[ignore]` is a libtest attribute this code cannot observe at
    /// runtime, so the default cannot key on it directly -- it keys on the
    /// family, which merely *correlates*.
    ///
    /// **That correlation is not an assumption left implicit.**
    /// `tests/skip_sites_are_opt_in.rs` scans every call site and fails if a
    /// `Fatal`-default family is used outside an `#[ignore]`d test. When the
    /// proxy stops holding, that guard says so.
    ///
    /// AOCL and MKL are `Permissive` for a measured reason rather than a
    /// symmetric one: `--features aocl` compiles from a sibling checkout with
    /// **no AMD DLLs present at all**, and the resulting test binary cannot even
    /// launch (`STATUS_DLL_NOT_FOUND`, measured on this box), so the probe it
    /// would gate never executes. For those families, compiling the feature
    /// carries no hardware implication whatsoever.
    pub const fn default_policy(self) -> Policy {
        match self {
            Hardware::Cuda | Hardware::Vulkan => Policy::Fatal,
            Hardware::Aocl | Hardware::Mkl => Policy::Permissive,
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

/// What happens when a device of some family is absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Absence is a test failure.
    Fatal,
    /// Absence is a declared skip.
    Permissive,
}

/// Resolve the policy from the family default and an environment override,
/// without touching the process environment.
///
/// Split out from [`skips_are_fatal`] so the reading rule is testable directly.
/// `std::env::set_var` is `unsafe` in edition 2024 and racy under the parallel
/// test harness, so a test that mutated the real environment would be both
/// unsound and flaky -- this function lets every branch be covered with neither.
///
/// Value semantics, and note that **`0` is not the same as unset**:
///
/// | `FUEL_REQUIRE_<FAM>` | result                       |
/// |----------------------|------------------------------|
/// | absent, or empty     | [`Hardware::default_policy`] |
/// | `0`                  | [`Policy::Permissive`]       |
/// | anything else        | [`Policy::Fatal`]            |
///
/// Empty resolves to the default rather than to `Fatal` because a shell that
/// exports every variable it names would otherwise arm every family at once.
/// `0` disarms **explicitly**, in both directions, so a wrong default is a
/// nuisance rather than a wall.
pub fn resolve_policy(hw: Hardware, value: Option<&OsStr>) -> Policy {
    match value {
        None => hw.default_policy(),
        Some(v) if v.is_empty() => hw.default_policy(),
        Some(v) if v == OsStr::new("0") => Policy::Permissive,
        Some(_) => Policy::Fatal,
    }
}

/// Whether device-absence for `hw` is currently fatal.
pub fn skips_are_fatal(hw: Hardware) -> bool {
    resolve_policy(hw, std::env::var_os(hw.require_var()).as_deref()) == Policy::Fatal
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

    /// Absent and empty fall through to the family default; `0` disarms.
    ///
    /// An earlier revision of this module treated ANY non-empty value as
    /// arming, so `FUEL_REQUIRE_CUDA=0` armed the gate. That was changed
    /// deliberately when per-family defaults landed: with a default of `Fatal`
    /// there has to be a way to say *no*, and `=0` is the obvious one. Recorded
    /// because it is a behaviour change, not a bug fix -- a script that set `=0`
    /// meaning "off" was previously getting "on".
    #[test]
    fn absent_and_empty_take_the_default_and_zero_disarms() {
        // Fatal-default family.
        assert_eq!(resolve_policy(Hardware::Cuda, None), Policy::Fatal);
        assert_eq!(
            resolve_policy(Hardware::Cuda, Some(OsStr::new(""))),
            Policy::Fatal,
            "empty must fall through to the default: a shell that exports every \
             variable it names must not change any family's behaviour"
        );
        assert_eq!(
            resolve_policy(Hardware::Cuda, Some(OsStr::new("0"))),
            Policy::Permissive,
            "`0` must disarm, or a wrong default is a wall instead of a nuisance"
        );

        // Permissive-default family: the override works in the other direction.
        assert_eq!(resolve_policy(Hardware::Aocl, None), Policy::Permissive);
        assert_eq!(
            resolve_policy(Hardware::Aocl, Some(OsStr::new("1"))),
            Policy::Fatal,
            "a machine that DOES have AOCL must be able to demand it"
        );
        assert_eq!(
            resolve_policy(Hardware::Aocl, Some(OsStr::new("0"))),
            Policy::Permissive
        );
    }

    /// The defaults are a deliberate split, not an accident of ordering.
    ///
    /// Pinned by family so that adding a variant forces a decision here rather
    /// than silently inheriting whichever arm a wildcard would have caught --
    /// `default_policy` has no wildcard, so a new family is a compile error.
    #[test]
    fn the_family_defaults_are_the_measured_split() {
        assert_eq!(Hardware::Cuda.default_policy(), Policy::Fatal);
        assert_eq!(Hardware::Vulkan.default_policy(), Policy::Fatal);
        assert_eq!(
            Hardware::Aocl.default_policy(),
            Policy::Permissive,
            "`--features aocl` compiles with no AMD DLLs present, so compiling \
             it implies nothing about hardware -- see `default_policy`'s docs"
        );
        assert_eq!(Hardware::Mkl.default_policy(), Policy::Permissive);
    }
}
