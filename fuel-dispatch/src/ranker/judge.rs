//! `JudgeOracle` — abstraction over the Phase 6b empirical
//! profiler for the cost composer's Layer-2 refinement.
//!
//! Phase 3 of the picker-work arc.
//!
//! # Why a trait
//!
//! The Phase 6b Judge + its `DispatchTable` + `ProfileReport` all
//! live in `fuel-core` (specifically `fuel_core::judge::*` after the
//! 2026-05-31 rename of `fuel_core::dispatch` → `fuel_core::judge`).
//! `fuel-dispatch` can't depend on `fuel-core` (that would cycle —
//! `fuel-core` already depends on `fuel-dispatch`), so the integration
//! happens through a trait: this module defines the contract, and
//! `fuel-core` ships an adapter impl on top of the live profile data.
//!
//! # The contract
//!
//! For each `(op, dtype, size_class, backend)` lookup, return the
//! measured median latency in nanoseconds when available. `None`
//! means "no measurement" — the cost composer leaves the Layer-1
//! static estimate in place rather than substituting a fabricated
//! number.
//!
//! Implementors are typically free-function HashMaps built once
//! from a `ProfileReport`. The trait stays narrow on purpose: the
//! optimizer ranker doesn't need to know HOW the Judge collected
//! the data, just that it can be queried per cell.

use fuel_core_types::dispatch::{OpKind, SizeClass};
use fuel_core_types::probe::BackendId;
use fuel_core_types::DType;

/// Read-only oracle over empirical measurements. Phase 3 wires this
/// into [`super::cost::compute_static_costs`] as the optional
/// Layer-2 refinement source.
pub trait JudgeOracle: Send + Sync {
    /// Median wall-clock latency in nanoseconds for the
    /// `(op, dtype, size_class, backend)` cell. Returns `None`
    /// when the cell isn't profiled. Callers MUST treat absence as
    /// "no measurement" — not "zero" — so the static estimate
    /// stays the fallback.
    fn measured_latency_ns(
        &self,
        op: OpKind,
        dtype: DType,
        size_class: SizeClass,
        backend: BackendId,
    ) -> Option<u64>;
}

/// HashMap-backed JudgeOracle. The simplest possible impl —
/// callers (and tests) populate it from a `ProfileReport` or
/// equivalent and hand it to [`super::cost::compute_static_costs`].
///
/// Production callers typically use this directly: `fuel_core`'s
/// integration computes a `HashMapJudge` from the cached
/// `ProfileReport` once at plan-start and feeds the same instance
/// to every `compile_plan` call until the topology generation
/// changes.
#[derive(Debug, Default, Clone)]
pub struct HashMapJudge {
    entries: std::collections::HashMap<(OpKind, DType, SizeClass, BackendId), u64>,
}

impl HashMapJudge {
    /// Empty map. Add entries via [`Self::insert`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert one measured latency. Replaces any prior entry with
    /// the same key (last-write-wins, mirroring the conventions in
    /// `fuel_core::judge::DispatchTable::rebuild_from`).
    pub fn insert(
        &mut self,
        op: OpKind,
        dtype: DType,
        size_class: SizeClass,
        backend: BackendId,
        latency_ns: u64,
    ) {
        self.entries
            .insert((op, dtype, size_class, backend), latency_ns);
    }

    /// Total number of populated cells.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is the map empty?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl JudgeOracle for HashMapJudge {
    fn measured_latency_ns(
        &self,
        op: OpKind,
        dtype: DType,
        size_class: SizeClass,
        backend: BackendId,
    ) -> Option<u64> {
        self.entries.get(&(op, dtype, size_class, backend)).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashmap_judge_round_trip() {
        let mut j = HashMapJudge::new();
        assert!(j.is_empty());
        j.insert(
            OpKind::MatMul,
            DType::F32,
            SizeClass(16),
            BackendId::Cuda,
            5_000_000,
        );
        assert_eq!(j.len(), 1);
        assert_eq!(
            j.measured_latency_ns(
                OpKind::MatMul,
                DType::F32,
                SizeClass(16),
                BackendId::Cuda,
            ),
            Some(5_000_000),
        );
    }

    #[test]
    fn hashmap_judge_miss_returns_none() {
        let j = HashMapJudge::new();
        assert!(j
            .measured_latency_ns(
                OpKind::AddElementwise,
                DType::F32,
                SizeClass(8),
                BackendId::Cpu,
            )
            .is_none());
    }

    #[test]
    fn hashmap_judge_last_write_wins() {
        let mut j = HashMapJudge::new();
        let key = (OpKind::MatMul, DType::F32, SizeClass(16), BackendId::Cpu);
        j.insert(key.0, key.1, key.2, key.3, 1_000);
        j.insert(key.0, key.1, key.2, key.3, 2_000);
        assert_eq!(
            j.measured_latency_ns(key.0, key.1, key.2, key.3),
            Some(2_000),
        );
    }

    #[test]
    fn hashmap_judge_distinguishes_backends_at_same_key() {
        let mut j = HashMapJudge::new();
        let (op, dt, sc) = (OpKind::MatMul, DType::F32, SizeClass(16));
        j.insert(op, dt, sc, BackendId::Cpu, 1_000_000);
        j.insert(op, dt, sc, BackendId::Cuda, 100_000);
        assert_eq!(j.measured_latency_ns(op, dt, sc, BackendId::Cpu), Some(1_000_000));
        assert_eq!(j.measured_latency_ns(op, dt, sc, BackendId::Cuda), Some(100_000));
        assert!(j.measured_latency_ns(op, dt, sc, BackendId::Vulkan).is_none());
    }

    #[test]
    fn trait_is_dyn_compatible() {
        // Compile-time check encoded as runtime construction.
        let j: Box<dyn JudgeOracle> = Box::new(HashMapJudge::new());
        assert!(j
            .measured_latency_ns(
                OpKind::AddElementwise,
                DType::F32,
                SizeClass(0),
                BackendId::Cpu,
            )
            .is_none());
    }
}
