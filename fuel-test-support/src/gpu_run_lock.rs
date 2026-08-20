//! **The GPU mutex's absence is indistinguishable from its success (GAP-224).**
//!
//! Every GPU-touching run is supposed to go through `scripts/gpu-run.ps1`, a
//! machine-wide named mutex that exists because of the 2026-07-31 host-aperture
//! kernel bugcheck (`docs/postmortems/`). Nothing enforced it. **A run that
//! skips the wrapper completes, passes, and differs from a guarded run only by
//! a lock nobody observes** — so the failure mode of the entire safety
//! mechanism is to disappear silently.
//!
//! That is not hypothetical and it did not need a broken wrapper as its excuse:
//! on 2026-08-20 a peer project reported, unprompted, running a workspace test
//! sweep that enumerates physical devices **with no wrapper at all**, chained
//! behind a `cargo fmt`, *with the rule written in their own durable notes*.
//! Nothing went wrong; the only reason anyone knows is that they said so.
//!
//! **So "remember the wrapper" is an instruction that decays, and decay was
//! demonstrated in someone who had the rule recorded.** The remedy is to make
//! the guarded thing detect the guard's absence.
//!
//! # How the lock is observed
//!
//! `gpu-run.ps1` already publishes everything needed, and none of it was read
//! by any Rust code before this module:
//!
//! - it exports `GPU_RUN_HELD=1` and `GPU_RUN_HELD_PID=<pid>` into the child
//!   environment (added for nested-invocation passthrough), and
//! - it writes `%TEMP%\gpu-run.lock` as compact JSON carrying that same pid,
//!   **and deletes the file in its `finally`**.
//!
//! Checking the environment variables *alone* would be defeated by the most
//! likely accident: `GPU_RUN_HELD=1` exported once into a long-lived debugging
//! shell satisfies it forever after, and that shell is exactly where someone
//! would set it. **Pairing them with the lockfile fixes that** — the file is
//! gone once the wrapper exits, so a stale export fails.
//!
//! This is deliberately preferred over checking that `GPU_RUN_HELD_PID` names a
//! *live process*: liveness answers *"is some process with that number
//! alive"*, which PID reuse can satisfy by accident, whereas the lockfile
//! answers *"is the lock held right now, by the process that exported this"* —
//! the question actually being asked — with no platform API and no dependency.
//!
//! # Two residuals, stated rather than smoothed
//!
//! 1. The `TEMP` this reads must be the `TEMP` the wrapper wrote to. Both use
//!    the process environment, so they agree in the normal case and can differ
//!    if a caller rewrites `TEMP` between them.
//! 2. A hard-killed holder can leave the file behind. A lingering lockfile
//!    beside a matching exported pid in a dead shell would pass. That window is
//!    narrow, `gpu-run.ps1` has its own stale-holder reclaim, and **a liveness
//!    check would not have escaped it either.**
//!
//! # This is a FAILURE, never a skip
//!
//! An unheld mutex is a misconfiguration of the **runner**, not a property of
//! the **machine**. Reporting it as a skip would make *"you forgot the
//! wrapper"* indistinguishable from *"this machine has no GPU"* — which is the
//! exact conflation this crate exists to prevent.

use std::path::PathBuf;

/// The wrapper invocation named in every failure, so the reader is told what to
/// run rather than merely what went wrong.
const WRAPPER: &str = "pwsh scripts/gpu-run.ps1 -Project <name> -- <cmd>";

/// Environment variable a caller sets to proceed **without** the lock.
///
/// Deliberately **not** `GPU_RUN_HELD`. If bypass meant setting the same
/// variable the wrapper sets, a hand-set value would be indistinguishable from
/// a real one and the defect would be rebuilt inside its own fix. It must also
/// carry a *reason*: writing a sentence is the cost that stops bypass becoming
/// habitual, and it makes a habitual bypass greppable afterwards.
pub const UNGUARDED_VAR: &str = "GPU_RUN_UNGUARDED";

/// What the environment and the lockfile say about the machine-wide GPU lock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuRunLock {
    /// The wrapper is holding the lock for this process tree.
    Held,
    /// The caller explicitly declared an unguarded run, with a reason.
    Unguarded {
        /// The caller's stated reason, echoed to stderr on every run.
        why: String,
    },
    /// No lock, and no declaration. `why` explains which check failed.
    Absent {
        /// Which specific check failed — not "no lock", but *why* we say so.
        why: String,
    },
}

/// Classify the lock from already-read inputs.
///
/// Split out from [`require_gpu_run_lock`] so the decision is testable without
/// mutating process-global environment state, which cannot be done safely from
/// tests that run in parallel.
///
/// `lock_contents` is the text of `gpu-run.lock`, or `None` when the file is
/// absent or unreadable — which are the same thing for this purpose: the
/// wrapper deletes it on release.
pub fn classify(
    held: Option<&str>,
    held_pid: Option<&str>,
    unguarded: Option<&str>,
    lock_contents: Option<&str>,
) -> GpuRunLock {
    // The declaration is checked FIRST and unconditionally: a caller who has
    // stated a reason gets the same answer whether or not a lock happens to be
    // held, so the printed line is a faithful record of what they asked for.
    if let Some(why) = unguarded {
        let why = why.trim();
        if !why.is_empty() {
            return GpuRunLock::Unguarded {
                why: why.to_string(),
            };
        }
        return GpuRunLock::Absent {
            why: format!("{UNGUARDED_VAR} is set but EMPTY — an unguarded run must state a reason"),
        };
    }

    if held != Some("1") {
        return GpuRunLock::Absent {
            why: "GPU_RUN_HELD is not \"1\" — this process was not started by the wrapper".into(),
        };
    }

    let Some(pid) = held_pid.map(str::trim).filter(|p| !p.is_empty()) else {
        return GpuRunLock::Absent {
            why: "GPU_RUN_HELD is \"1\" but GPU_RUN_HELD_PID is unset — the environment is \
                  half-set, which a hand-export produces and the wrapper never does"
                .into(),
        };
    };

    // The lockfile is what makes a stale environment export fail: `gpu-run.ps1`
    // removes it in its `finally`, so its presence means the lock is held NOW,
    // and the pid match means it is held by the process that exported these.
    let Some(contents) = lock_contents else {
        return GpuRunLock::Absent {
            why: format!(
                "GPU_RUN_HELD=1 and GPU_RUN_HELD_PID={pid}, but gpu-run.lock is absent — the \
                 wrapper deletes it on release, so these variables are a STALE EXPORT from a \
                 shell that outlived its run"
            ),
        };
    };

    if !names_pid(contents, pid) {
        return GpuRunLock::Absent {
            why: format!(
                "gpu-run.lock exists but does not name pid {pid} — the lock is held by a \
                 DIFFERENT run, so this process is not the holder"
            ),
        };
    }

    GpuRunLock::Held
}

/// Does `contents` carry `"pid":<pid>` as a whole number?
///
/// ⚠️ A plain `contains("\"pid\":{pid}")` is WRONG and the test above caught it:
/// pid `424` matches a lockfile holding `4242`, because the needle is a prefix
/// of the haystack's number. **A delimiter trap has two ends** — the `"pid":`
/// prefix is a delimiter and the digits' end is not, so the match must be
/// terminated as well as anchored. The failure direction is the bad one: a
/// *different* run's lock would be accepted as ours.
fn names_pid(contents: &str, pid: &str) -> bool {
    let needle = format!("\"pid\":{pid}");
    let mut from = 0usize;
    while let Some(at) = contents[from..].find(&needle) {
        let end = from + at + needle.len();
        match contents[end..].chars().next() {
            Some(c) if c.is_ascii_digit() => from = end, // a longer number — keep looking
            _ => return true,
        }
    }
    false
}

/// Path the wrapper writes its metadata lockfile to.
fn lock_path() -> Option<PathBuf> {
    std::env::var_os("TEMP").map(|t| PathBuf::from(t).join("gpu-run.lock"))
}

/// Require that this process is running under the machine-wide GPU lock.
///
/// Panics with the wrapper command when it is not. Prints
/// `PROCEEDING UNGUARDED: <why>` to stderr — **on every run, never once** —
/// when [`UNGUARDED_VAR`] carries a reason: an escape hatch that produces no
/// output recreates the exact defect this exists to remove, because the whole
/// problem is that absence looks like success.
///
/// `#[track_caller]` points the panic at the test that needed the device.
#[track_caller]
pub fn require_gpu_run_lock() {
    let held = std::env::var("GPU_RUN_HELD").ok();
    let pid = std::env::var("GPU_RUN_HELD_PID").ok();
    let unguarded = std::env::var(UNGUARDED_VAR).ok();
    let contents = lock_path().and_then(|p| std::fs::read_to_string(p).ok());

    match classify(
        held.as_deref(),
        pid.as_deref(),
        unguarded.as_deref(),
        contents.as_deref(),
    ) {
        GpuRunLock::Held => {}
        GpuRunLock::Unguarded { why } => {
            eprintln!("PROCEEDING UNGUARDED: {why}");
        }
        GpuRunLock::Absent { why } => panic!(
            "this test touches the GPU and the machine-wide gpu-run lock is NOT held ({why}).\n\
             Run it through the wrapper:\n    {WRAPPER}\n\
             Skipping the wrapper is what the 2026-07-31 host-aperture bugcheck came from, and \
             an unguarded run is indistinguishable from a guarded one in every output except \
             this message. If you must proceed without it, set {UNGUARDED_VAR}=<reason> — the \
             reason is printed on every run so a habitual bypass stays visible."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCK: &str = r#"{"v":1,"sv":"1.4.0","pid":4242,"project":"fuel","cmd":"cargo test","start":"2026-08-20T18:00:00.0000000+00:00"}"#;

    /// NEGATIVE CONTROL. Without an arm that returns `Held`, every assertion
    /// below is satisfied by a function that refuses unconditionally.
    #[test]
    fn a_real_wrapper_run_is_held() {
        assert_eq!(
            classify(Some("1"), Some("4242"), None, Some(LOCK)),
            GpuRunLock::Held
        );
    }

    #[test]
    fn nothing_set_is_absent() {
        assert!(matches!(
            classify(None, None, None, None),
            GpuRunLock::Absent { .. }
        ));
    }

    /// The defect the lockfile exists to catch: `GPU_RUN_HELD=1` exported once
    /// into a shell that outlived its run.
    #[test]
    fn a_stale_environment_export_is_absent() {
        let got = classify(Some("1"), Some("4242"), None, None);
        let GpuRunLock::Absent { why } = got else {
            panic!("expected Absent, got {got:?}")
        };
        assert!(
            why.contains("STALE EXPORT"),
            "the reason must name the mechanism: {why}"
        );
    }

    /// DISCRIMINATION. A lockfile held by a *different* run must not satisfy
    /// this process — otherwise the check degrades to "some GPU run exists".
    #[test]
    fn a_lock_held_by_another_run_is_absent() {
        let got = classify(Some("1"), Some("9999"), None, Some(LOCK));
        let GpuRunLock::Absent { why } = got else {
            panic!("expected Absent, got {got:?}")
        };
        assert!(
            why.contains("DIFFERENT run"),
            "the reason must name the mechanism: {why}"
        );
    }

    /// Substring matching must not accept a pid that merely *contains* ours.
    ///
    /// This test FAILED on the first implementation, which used a plain
    /// `contains("\"pid\":{pid}")`: pid `424` matched a lockfile holding
    /// `4242`. Kept with both ends exercised, because a delimiter trap has two
    /// of them and fixing the prefix is what makes the suffix easy to forget.
    #[test]
    fn a_pid_substring_does_not_match() {
        // Ours is a PREFIX of the holder's number.
        assert!(matches!(
            classify(Some("1"), Some("424"), None, Some(LOCK)),
            GpuRunLock::Absent { .. }
        ));
        // Ours is a SUFFIX of the holder's number.
        assert!(matches!(
            classify(Some("1"), Some("242"), None, Some(LOCK)),
            GpuRunLock::Absent { .. }
        ));
        // And the whole number still matches when it is the LAST field, where
        // there is no trailing comma to lean on.
        assert_eq!(
            classify(
                Some("1"),
                Some("4242"),
                None,
                Some(r#"{"project":"fuel","pid":4242}"#)
            ),
            GpuRunLock::Held
        );
    }

    #[test]
    fn half_set_environment_is_absent() {
        let got = classify(Some("1"), None, None, Some(LOCK));
        let GpuRunLock::Absent { why } = got else {
            panic!("expected Absent, got {got:?}")
        };
        assert!(
            why.contains("half-set"),
            "the reason must name the mechanism: {why}"
        );
    }

    #[test]
    fn an_explicit_declaration_proceeds_and_carries_its_reason() {
        assert_eq!(
            classify(None, None, Some("bisecting a driver hang"), None),
            GpuRunLock::Unguarded {
                why: "bisecting a driver hang".into()
            }
        );
    }

    /// An empty declaration is not a declaration. Allowing `GPU_RUN_UNGUARDED=`
    /// to work would give back the costless bypass the reason exists to prevent.
    #[test]
    fn an_empty_declaration_is_not_a_declaration() {
        assert!(matches!(
            classify(None, None, Some("   "), None),
            GpuRunLock::Absent { .. }
        ));
    }
}
