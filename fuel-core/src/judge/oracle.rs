// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ProfileJudgeOracle` — fuel-core's concrete
//! [`JudgeOracle`](fuel_dispatch::ranker::JudgeOracle) over the
//! Judge's measured profile data.
//!
//! Phase 3 of the picker-work arc shipped the `JudgeOracle` trait +
//! `PlanOptions::with_judge` in `fuel-dispatch`, but `fuel-dispatch`
//! can't depend on `fuel-core` (dependency cycle), so the adapter
//! over the real measurement store lives here.
//!
//! # Data source: `ProfileReport`, not `DispatchTable`
//!
//! The Judge produces two artifacts:
//!
//! - [`ProfileReport`] — every raw measurement: one entry per
//!   `(op, dtype, size_class, backend, device, kernel_source)` cell,
//!   including LOSING alternatives (per-alternative measurement,
//!   commit `1ba99650`).
//! - `DispatchTable` — winners only, indexed per criterion.
//!
//! The cost composer's Layer-2 refinement needs latencies for EVERY
//! candidate it ranks — a losing-but-close alternative must carry its
//! own measured number, not inherit the winner's. So the adapter is
//! built from the report, not the table.
//!
//! # Key semantics
//!
//! Exact-match on all five axes. `kernel_source` is part of the key:
//! AOCL / MKL / portable-cpu siblings at the same
//! `(op, dtype, size_class, BackendId::Cpu)` cell each resolve to
//! their own latency, and an unmeasured sibling misses (`None`)
//! rather than borrowing a sibling's number.
//!
//! The trait key carries no device axis; when a report holds the same
//! cell measured on multiple devices (distinct equivalence classes of
//! one backend), the adapter keeps the MINIMUM latency — "the best
//! this backend has demonstrated" — which is deterministic regardless
//! of entry order. Within one equivalence class the replicated
//! entries share a latency, so the min is a no-op there.

use fuel_dispatch::ranker::{HashMapJudge, JudgeOracle};
use fuel_ir::DType;
use fuel_ir::dispatch::{OpKind, PROFILE_REPORT_VERSION, ProfileReport, SizeClass};
use fuel_ir::probe::BackendId;

/// [`JudgeOracle`] adapter over a [`ProfileReport`]. Build once per
/// report (cache lifecycle in [`super::cache`]), query per candidate
/// from the optimizer ranker's cost composer.
#[derive(Debug, Default, Clone)]
pub struct ProfileJudgeOracle {
    inner: HashMapJudge,
}

impl ProfileJudgeOracle {
    /// Index every entry of `report` for exact-match lookup.
    ///
    /// Defensive version gate: a report whose `version` differs from
    /// [`PROFILE_REPORT_VERSION`] produces an EMPTY oracle (every
    /// lookup misses → Layer-1 static costs stand). The persisted-load
    /// path already filters stale schemas (`ProfileReport::load`
    /// returns `Ok(None)`), but an in-memory report of a foreign
    /// schema must not feed the cost composer either — pre-v2 reports
    /// were ambiguous about which kernel sibling they timed.
    pub fn from_report(report: &ProfileReport) -> Self {
        let mut inner = HashMapJudge::new();
        if report.version != PROFILE_REPORT_VERSION {
            return Self { inner };
        }
        for e in &report.entries {
            // `device.backend` is a redundant co-encoding of the `backend`
            // axis; a divergence means a producer stamped a device identity
            // that disagrees with the backend it measured on. Nothing downstream
            // re-derives one from the other, so this is the only thing keeping
            // the two in step — catch a producer bug at its source, in tests.
            debug_assert_eq!(
                e.backend, e.device.backend,
                "ProfileEntry backend ({:?}) and device.backend ({:?}) disagree",
                e.backend, e.device.backend
            );
            let prev = inner.measured_latency_ns(
                e.op,
                e.dtype,
                e.size_class,
                e.backend,
                &e.kernel_source,
                e.kernel_revision_hash,
            );
            // Keep the minimum across GENUINE duplicate keys — multi-device
            // entries at ONE `(op, dtype, size_class, backend, kernel_source,
            // kernel_revision_hash)` cell (see module docs). Distinct
            // revisions are distinct keys, so variant siblings no longer
            // collapse here: they land in separate cells and each keeps its
            // own measured latency.
            if prev.is_none_or(|p| e.latency_ns < p) {
                inner.insert(
                    e.op,
                    e.dtype,
                    e.size_class,
                    e.backend,
                    &e.kernel_source,
                    e.latency_ns,
                    e.kernel_revision_hash,
                );
            }
        }
        Self { inner }
    }

    /// Number of distinct `(op, dtype, size_class, backend,
    /// kernel_source)` cells indexed.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Is the oracle empty (no measurements — every lookup misses)?
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl JudgeOracle for ProfileJudgeOracle {
    fn measured_latency_ns(
        &self,
        op: OpKind,
        dtype: DType,
        size_class: SizeClass,
        backend: BackendId,
        kernel_source: &str,
        kernel_revision_hash: u64,
    ) -> Option<u64> {
        self.inner.measured_latency_ns(
            op,
            dtype,
            size_class,
            backend,
            kernel_source,
            kernel_revision_hash,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::test_equiv_key;
    use fuel_ir::dispatch::ProfileEntry;

    fn entry(
        backend: BackendId,
        op: OpKind,
        size: u32,
        device_index: u32,
        latency: u64,
        kernel_source: &str,
    ) -> ProfileEntry {
        ProfileEntry {
            op,
            dtype: DType::F32,
            size_class: SizeClass(size),
            backend,
            device_index,
            latency_ns: latency,
            iterations: 7,
            max_rel_error: 1e-6,
            kernel_source: kernel_source.to_string(),
            kernel_revision_hash: 0,
            device: test_equiv_key(backend),
        }
    }

    #[test]
    fn empty_report_misses_everywhere() {
        let report = ProfileReport {
            version: PROFILE_REPORT_VERSION,
            entries: vec![],
        };
        let oracle = ProfileJudgeOracle::from_report(&report);
        assert!(oracle.is_empty());
        assert!(
            oracle
                .measured_latency_ns(
                    OpKind::MatMul,
                    DType::F32,
                    SizeClass(16),
                    BackendId::Cpu,
                    "",
                    0,
                )
                .is_none()
        );
    }

    /// Two entries at the same `(op, dtype, size_class, backend)`
    /// cell differing ONLY in `kernel_source` resolve to distinct
    /// latencies — the sibling-collision bug class the per-alt
    /// remediation arc fixed must stay fixed here.
    #[test]
    fn sibling_kernel_sources_do_not_collide() {
        let report = ProfileReport {
            version: PROFILE_REPORT_VERSION,
            entries: vec![
                entry(
                    BackendId::Cpu,
                    OpKind::MatMul,
                    16,
                    0,
                    1_000_000,
                    "portable-cpu",
                ),
                entry(BackendId::Cpu, OpKind::MatMul, 16, 0, 200_000, "aocl"),
            ],
        };
        let oracle = ProfileJudgeOracle::from_report(&report);
        assert_eq!(oracle.len(), 2);
        let cell = |src: &str| {
            oracle.measured_latency_ns(
                OpKind::MatMul,
                DType::F32,
                SizeClass(16),
                BackendId::Cpu,
                src,
                0,
            )
        };
        assert_eq!(cell("portable-cpu"), Some(1_000_000));
        assert_eq!(cell("aocl"), Some(200_000));
        // An unmeasured sibling misses — no fallthrough to either
        // measured sibling.
        assert!(cell("mkl").is_none());
        assert!(cell("").is_none());
    }

    /// Same key measured on two devices (distinct equivalence
    /// classes of one backend) keeps the minimum, independent of
    /// entry order.
    #[test]
    fn duplicate_keys_keep_minimum_latency() {
        let report = ProfileReport {
            version: PROFILE_REPORT_VERSION,
            entries: vec![
                entry(BackendId::Cuda, OpKind::MatMul, 20, 0, 300_000, "cublas"),
                entry(BackendId::Cuda, OpKind::MatMul, 20, 1, 900_000, "cublas"),
            ],
        };
        let oracle = ProfileJudgeOracle::from_report(&report);
        assert_eq!(oracle.len(), 1);
        assert_eq!(
            oracle.measured_latency_ns(
                OpKind::MatMul,
                DType::F32,
                SizeClass(20),
                BackendId::Cuda,
                "cublas",
                0,
            ),
            Some(300_000),
        );

        // Reverse the entry order — same answer.
        let reversed = ProfileReport {
            version: PROFILE_REPORT_VERSION,
            entries: vec![
                entry(BackendId::Cuda, OpKind::MatMul, 20, 1, 900_000, "cublas"),
                entry(BackendId::Cuda, OpKind::MatMul, 20, 0, 300_000, "cublas"),
            ],
        };
        let oracle = ProfileJudgeOracle::from_report(&reversed);
        assert_eq!(
            oracle.measured_latency_ns(
                OpKind::MatMul,
                DType::F32,
                SizeClass(20),
                BackendId::Cuda,
                "cublas",
                0,
            ),
            Some(300_000),
        );
    }

    /// Misses stay misses on every non-matching axis: backend,
    /// dtype, size_class, op.
    #[test]
    fn miss_on_any_axis_returns_none() {
        let report = ProfileReport {
            version: PROFILE_REPORT_VERSION,
            entries: vec![entry(BackendId::Cpu, OpKind::MatMul, 16, 0, 1_000, "")],
        };
        let oracle = ProfileJudgeOracle::from_report(&report);
        let hit = oracle.measured_latency_ns(
            OpKind::MatMul,
            DType::F32,
            SizeClass(16),
            BackendId::Cpu,
            "",
            0,
        );
        assert_eq!(hit, Some(1_000));
        assert!(
            oracle
                .measured_latency_ns(
                    OpKind::MatMul,
                    DType::F32,
                    SizeClass(16),
                    BackendId::Cuda,
                    "",
                    0,
                )
                .is_none()
        );
        assert!(
            oracle
                .measured_latency_ns(
                    OpKind::MatMul,
                    DType::F64,
                    SizeClass(16),
                    BackendId::Cpu,
                    "",
                    0,
                )
                .is_none()
        );
        assert!(
            oracle
                .measured_latency_ns(
                    OpKind::MatMul,
                    DType::F32,
                    SizeClass(17),
                    BackendId::Cpu,
                    "",
                    0,
                )
                .is_none()
        );
        assert!(
            oracle
                .measured_latency_ns(
                    OpKind::AddElementwise,
                    DType::F32,
                    SizeClass(16),
                    BackendId::Cpu,
                    "",
                    0,
                )
                .is_none()
        );
    }

    /// A report with a foreign schema version produces an empty
    /// oracle — Layer-1 static costs stand.
    #[test]
    fn version_mismatch_yields_empty_oracle() {
        let report = ProfileReport {
            version: PROFILE_REPORT_VERSION + 1,
            entries: vec![entry(BackendId::Cpu, OpKind::MatMul, 16, 0, 1_000, "")],
        };
        let oracle = ProfileJudgeOracle::from_report(&report);
        assert!(oracle.is_empty());
    }

    /// Type-level proof of the production plumb: the adapter slots
    /// into `PlanOptions::with_judge` as a `&dyn JudgeOracle`.
    #[test]
    fn adapter_attaches_to_plan_options() {
        let report = ProfileReport {
            version: PROFILE_REPORT_VERSION,
            entries: vec![entry(BackendId::Cpu, OpKind::MatMul, 16, 0, 1_000, "aocl")],
        };
        let oracle = ProfileJudgeOracle::from_report(&report);
        let opts = fuel_dispatch::plan::PlanOptions::new().with_judge(&oracle);
        let queried = opts
            .judge
            .expect("with_judge sets the oracle")
            .measured_latency_ns(
                OpKind::MatMul,
                DType::F32,
                SizeClass(16),
                BackendId::Cpu,
                "aocl",
                0,
            );
        assert_eq!(queried, Some(1_000));
    }

    /// Two entries at the SAME `(op, dtype, size, backend, kernel_source)`
    /// cell differing ONLY in `kernel_revision_hash` — both NON-ZERO — occupy
    /// TWO distinct oracle cells via the REAL cache path (`from_report`).
    /// This is the variant-sibling collapse `kernel_revision_hash` exists to
    /// prevent: forty tilings of one provider share the `kernel_source` tag
    /// (`"baracuda"`), so without the revision in the key they collapse into
    /// one cell and thirty-nine become invisible with no error. Both revisions
    /// are nonzero deliberately — a `0`-vs-nonzero test would pass while real
    /// siblings still collapsed. Born-red: before the revision joined the key
    /// this produced `len() == 1` and both lookups returned the min latency.
    #[test]
    fn sibling_kernel_revisions_do_not_collide_via_report() {
        let mk = |rev: u64, latency: u64| ProfileEntry {
            op: OpKind::MatMul,
            dtype: DType::F32,
            size_class: SizeClass(16),
            backend: BackendId::Cuda,
            device_index: 0,
            latency_ns: latency,
            iterations: 7,
            max_rel_error: 1e-6,
            kernel_source: "baracuda".to_string(),
            kernel_revision_hash: rev,
            device: test_equiv_key(BackendId::Cuda),
        };
        let report = ProfileReport {
            version: PROFILE_REPORT_VERSION,
            entries: vec![
                mk(0x1111_1111_1111_1111, 5_000),
                mk(0x2222_2222_2222_2222, 9_000),
            ],
        };
        let oracle = ProfileJudgeOracle::from_report(&report);
        assert_eq!(
            oracle.len(),
            2,
            "distinct non-zero revisions are distinct cells"
        );
        let cell = |rev: u64| {
            oracle.measured_latency_ns(
                OpKind::MatMul,
                DType::F32,
                SizeClass(16),
                BackendId::Cuda,
                "baracuda",
                rev,
            )
        };
        assert_eq!(cell(0x1111_1111_1111_1111), Some(5_000));
        assert_eq!(cell(0x2222_2222_2222_2222), Some(9_000));
        // An untracked (0) revision at the same tag is its OWN cell — no
        // fallthrough to either measured sibling.
        assert!(cell(0).is_none());
    }
}
