// SPDX-License-Identifier: MIT OR Apache-2.0
//! Strict reproducibility mode (ROADMAP item 13(b)).
//!
//! A *policy* surface over substrate that already exists: every registered
//! kernel declares `PrecisionGuarantee::bit_stable_on_same_hardware`, and
//! `fuel_dispatch`'s `PrecisionFloorFilter` is a HARD filter that refuses any
//! candidate whose claim does not meet a `PrecisionRequirement`. What was
//! missing is a way for a caller to reach it: the production realize path
//! (`pipelined_bridge::build_optimized_graph`) never set
//! `PlanOptions::precision_requirement`, so the default chain's floor was
//! always unconstrained.
//!
//! Under strict mode the planner refuses every non-bit-stable arm. A node
//! whose only arms are non-bit-stable on the pinned device falls back to an
//! admissible off-device sibling when one is feasible; otherwise planning
//! fails with `Error::FilterRejected { filter: "precision-floor", .. }` —
//! loudly, at graph-build time, never by silently picking a non-reproducible
//! kernel.
//!
//! "Bit-stable" here is the declared claim: repeat runs on the SAME hardware
//! produce identical bytes. It is not a cross-hardware guarantee (see
//! `docs/architecture/05-backend-contract.md`).
//!
//! # Scope
//!
//! The mode is read where the plan is BUILT, on the realize caller's thread.
//! A scoped guard ([`strict_reproducibility`]) covers one thread and nests;
//! [`set_global_strict_reproducibility`] covers the process. Strict wins if
//! either is on.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

use fuel_dispatch::PrecisionRequirement;

static GLOBAL_STRICT: AtomicBool = AtomicBool::new(false);

std::thread_local! {
    static SCOPE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// RAII guard returned by [`strict_reproducibility`]. Strict mode stays on for
/// this thread until every outstanding guard is dropped. Not `Send`: the
/// scope belongs to the thread that opened it.
#[must_use = "strict mode ends when the guard is dropped"]
pub struct StrictReproducibilityGuard {
    _not_send: PhantomData<*const ()>,
}

/// Enter strict reproducibility mode on the current thread until the returned
/// guard is dropped. Nests.
pub fn strict_reproducibility() -> StrictReproducibilityGuard {
    SCOPE_DEPTH.with(|d| d.set(d.get() + 1));
    StrictReproducibilityGuard {
        _not_send: PhantomData,
    }
}

impl Drop for StrictReproducibilityGuard {
    fn drop(&mut self) {
        SCOPE_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Turn strict mode on or off for the whole process. Composes with scoped
/// guards: either one being on makes [`is_strict_reproducibility`] true.
pub fn set_global_strict_reproducibility(on: bool) {
    GLOBAL_STRICT.store(on, Ordering::Relaxed);
}

/// Is strict reproducibility in force on the current thread right now?
pub fn is_strict_reproducibility() -> bool {
    GLOBAL_STRICT.load(Ordering::Relaxed) || SCOPE_DEPTH.with(|d| d.get() > 0)
}

/// The precision floor the realize path must hand to the planner.
pub(crate) fn precision_requirement() -> PrecisionRequirement {
    if is_strict_reproducibility() {
        PrecisionRequirement::BIT_STABLE
    } else {
        PrecisionRequirement::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_by_default_is_unconstrained() {
        assert!(!is_strict_reproducibility());
        assert!(precision_requirement().is_unconstrained());
    }

    #[test]
    fn scope_requires_bit_stable_and_restores_on_drop() {
        {
            let _g = strict_reproducibility();
            assert!(is_strict_reproducibility());
            assert_eq!(precision_requirement(), PrecisionRequirement::BIT_STABLE);
        }
        assert!(!is_strict_reproducibility());
        assert!(precision_requirement().is_unconstrained());
    }

    #[test]
    fn scopes_nest() {
        let outer = strict_reproducibility();
        {
            let _inner = strict_reproducibility();
        }
        assert!(is_strict_reproducibility(), "inner drop must not end outer");
        drop(outer);
        assert!(!is_strict_reproducibility());
    }

    #[test]
    fn scope_is_thread_local() {
        let _g = strict_reproducibility();
        let other = std::thread::spawn(is_strict_reproducibility)
            .join()
            .unwrap();
        assert!(!other, "a scope must not leak to other threads");
    }
}
