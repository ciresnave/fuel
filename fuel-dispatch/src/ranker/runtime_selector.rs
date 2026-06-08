//! `RuntimeSelector` — Picker 2 substrate.
//!
//! Phase 5.1 of the picker-work arc. The optimizer ranker (Picker
//! 1, Phases 1.1–1.5 + 3) commits to a per-decision-point top-N
//! [`AlternativeSet`]; the runtime selector (this module) picks
//! among them at dispatch time.
//!
//! # Why two pickers
//!
//! Per architecture v1.0 §04 and the 2026-05-30 picker-alternatives
//! audit, the architectural endpoint is two distinct pick surfaces:
//!
//! - **Picker 1 — the optimizer ranker.** Plan-time. Enumerates
//!   candidates via SystemTopology, hard-filters by correctness
//!   (precision floor, capability), ranks by composite cost (Layer-1
//!   static + Layer-2 Judge data), preserves top-N. The output is
//!   the [`crate::plan::ExecutionPlan`] consumed by the executor.
//!
//! - **Picker 2 — the runtime selector (this trait).** Dispatch-time.
//!   Receives the plan's top-N for a node, picks one based on
//!   layer-3 signals (device load, recent failures, fresh
//!   telemetry) that didn't exist at plan time.
//!
//! Picker 1 commits to a *ranking*; Picker 2 commits to a *pick*.
//! The seam between them lets Picker 1 cache plans across realizes
//! while Picker 2's choices reflect current state at the moment of
//! dispatch.
//!
//! # What ships in 5.1
//!
//! Just the trait surface + [`WinnerSelector`] (the static default —
//! returns `set.winner()` = the top-N's first entry = the static-
//! cost winner). Behaviorally identical to pre-5.1 dispatch.
//!
//! # What 5.2+ will add
//!
//! Concrete selectors that exploit runtime signals. Likely
//! candidates (sketched, not committed):
//!
//! - `JudgeAwareSelector` — re-query the Judge at dispatch time
//!   for any candidate whose Layer-2 measurement landed after the
//!   plan's snapshot.
//! - `SizeAwareDownscaleSelector` — for tiny ops, prefer CPU over
//!   GPU to avoid kernel-launch overhead even when GPU is the
//!   static-cost winner.
//! - `FailureMemorySelector` — per-(op, backend) recent-failure
//!   tracking; demote a backend in the top-N after a transient
//!   error.
//! - `DeviceLoadSelector` — queue-depth + stream-utilization
//!   probing; spread work across co-located backends.
//!
//! None of these are wired today — each needs telemetry
//! infrastructure that doesn't exist yet. The seam lets them ship
//! independently when the signals do.

use crate::ranker::{AlternativeSet, Candidate};

/// Dispatch-time selector over a plan's per-node [`AlternativeSet`].
/// Phase 5.1 of the picker-work arc.
///
/// # Contract
///
/// - `select` is called exactly once per kernel-bearing node, at
///   dispatch time, by the executor's `resolve_compiled`.
/// - Implementors return `None` only when the set is empty — every
///   non-empty set MUST have a pick. The default implementation
///   ([`WinnerSelector`]) returns `set.winner()` (the first entry).
/// - The selected `Candidate`'s `kernel`, `caps`, and `backend`
///   become the executor's dispatch parameters for that node.
/// - Selectors are queried per-node and may make different picks
///   for different nodes in the same realize — the trait carries
///   no per-realize state.
///
/// # Sendness
///
/// Implementations must be `Send + Sync` so the executor can share
/// a single selector across the compiler thread and the executor
/// thread.
///
/// # Why not pass NodeId / OpKind / dtypes?
///
/// The [`Candidate`] already carries `backend`, `precision`,
/// `static_cost`, and `op_params`; selectors that need op identity
/// can derive it from these fields. If a future selector needs the
/// graph-level NodeId or OpKind, we add a [`SelectorContext`]
/// param — but Phase 5.1 keeps the surface minimal so the trait
/// stays usable by selectors that only care about set contents.
pub trait RuntimeSelector: Send + Sync + std::fmt::Debug {
    /// Pick one candidate from the set. `None` only when the set is
    /// empty; non-empty sets MUST produce a pick.
    ///
    /// The default implementation returns `set.winner()` — the top-N's
    /// first entry, which is Picker 1's static-cost winner after
    /// rank+truncate.
    fn select<'a>(&self, set: &'a AlternativeSet) -> Option<&'a Candidate> {
        set.winner()
    }
}

/// The trivial selector: always returns `set.winner()`. Behaviorally
/// identical to pre-Phase-5 dispatch — every call that doesn't
/// explicitly supply a selector falls back to this.
///
/// Implements [`RuntimeSelector`] via the trait's default `select`.
/// Use directly via `WinnerSelector` or wrap in
/// `Arc::new(WinnerSelector) as Arc<dyn RuntimeSelector>` for the
/// executor APIs that take an `Arc<dyn RuntimeSelector>`.
#[derive(Debug, Default, Clone, Copy)]
pub struct WinnerSelector;

impl RuntimeSelector for WinnerSelector {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::{CostEstimate, PrecisionGuarantee};
    use crate::kernel::{KernelCaps, OpParams};
    use fuel_core_types::probe::BackendId;
    use fuel_core_types::{DeviceLocation, Layout, Result};
    use fuel_storage::Storage;
    use std::sync::{Arc, RwLock};

    fn noop_kernel(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn make_candidate(backend: BackendId, cost_ns: u64) -> Candidate {
        Candidate {
            kernel: noop_kernel,
            caps: KernelCaps::empty(),
            backend,
            device: DeviceLocation::Cpu,
            precision: PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            static_cost: CostEstimate {
                flops: cost_ns,
                bytes_moved: cost_ns,
                kernel_overhead_ns: 0,
            },
            op_params: OpParams::None,
            coupling: Vec::new(),
            kernel_source: "",
        }
    }

    /// WinnerSelector returns the set's first entry — the static-
    /// cost winner. Behavioral baseline.
    #[test]
    fn winner_selector_returns_set_winner() {
        let mut set = AlternativeSet::empty();
        set.push(make_candidate(BackendId::Cuda, 100));
        set.push(make_candidate(BackendId::Cpu, 200));
        let selector = WinnerSelector;
        let pick = selector.select(&set).expect("non-empty set");
        assert_eq!(pick.backend, BackendId::Cuda);
    }

    /// WinnerSelector on empty set returns None — matches
    /// `AlternativeSet::winner` semantics.
    #[test]
    fn winner_selector_empty_set_returns_none() {
        let set = AlternativeSet::empty();
        let selector = WinnerSelector;
        assert!(selector.select(&set).is_none());
    }

    /// Custom impl can override the default and pick non-winner.
    /// This proves the trait is a real seam — a future Picker 2
    /// implementation can demote the static winner based on its
    /// own logic.
    #[test]
    fn custom_selector_can_pick_non_winner() {
        /// Picks the candidate with the LAST backend in the set
        /// (opposite of winner-first). Test-only.
        #[derive(Debug)]
        struct LastEntrySelector;
        impl RuntimeSelector for LastEntrySelector {
            fn select<'a>(&self, set: &'a AlternativeSet) -> Option<&'a Candidate> {
                set.alternatives().last()
            }
        }

        let mut set = AlternativeSet::empty();
        set.push(make_candidate(BackendId::Cuda, 100));
        set.push(make_candidate(BackendId::Vulkan, 200));
        set.push(make_candidate(BackendId::Cpu, 300));

        let pick = LastEntrySelector.select(&set).expect("non-empty");
        assert_eq!(
            pick.backend, BackendId::Cpu,
            "custom selector overrides default winner-pick",
        );
        // And the default would have picked AOCL — proves they
        // disagree.
        assert_eq!(WinnerSelector.select(&set).unwrap().backend, BackendId::Cuda);
    }

    /// Trait is dyn-compatible so the executor can pass
    /// `&dyn RuntimeSelector` / `Arc<dyn RuntimeSelector>`.
    #[test]
    fn trait_is_dyn_compatible() {
        let boxed: Box<dyn RuntimeSelector> = Box::new(WinnerSelector);
        let set = AlternativeSet::empty();
        assert!(boxed.select(&set).is_none());
    }
}
