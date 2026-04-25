//! Ranked dispatch tables — Phase 6b's O(1) runtime lookup.
//!
//! The [`Judge`](crate::judge) produces a [`ProfileReport`]
//! ([`crate::judge::ProfileReport`]) of raw measurements. That's
//! not yet useful for realize-time dispatch — the router needs
//! to ask "given this (op, dtype, size_class), which backend wins
//! under criterion X?" and get a constant-time answer. This
//! module translates the Judge's raw matrix into that lookup
//! table.
//!
//! # Criteria
//!
//! - [`Criterion::Fastest`] — pick the backend with the lowest
//!   median latency. Excludes the reference backend by default
//!   (it's the oracle, not a production path — pathologically
//!   slow by design).
//! - [`Criterion::MostAccurate`] — pick the backend with the lowest
//!   `max_rel_error` against the reference backend. Also excludes
//!   reference (reference vs reference is by definition 0, but
//!   dispatching to it defeats the purpose of having an optimized
//!   backend at all). Ties broken by latency.
//! - [`Criterion::Balanced`] — minimise `latency_ns * (1 +
//!   accuracy_penalty * rel_error)`. Penalty coefficient defaults to
//!   100 so a 1% rel-error bump is equivalent to a 2× latency
//!   penalty — steep enough that numerically-unsound fast paths
//!   don't win by default.
//!
//! # Fallback for unprofiled sizes
//!
//! The Judge only profiles a discrete set of size classes. A real
//! dispatch query at runtime arrives with a specific shape; if that
//! shape's log2-bucket wasn't measured, we fall back to the nearest
//! profiled class. Near-neighbour lookup is in [`DispatchTable::pick_nearest`].
//!
//! # Reference backend
//!
//! By default the reference backend is excluded from dispatch picks
//! — it's there as a correctness oracle, not a production executor.
//! [`DispatchTable::with_reference_backend`] opts it back in if
//! you want to force correctness-at-any-cost for a particular path
//! (debugging, say).

use crate::judge::{OpKind, ProfileEntry, ProfileReport, SizeClass};
use fuel_core_types::probe::BackendId;
use fuel_core_types::DType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A selection criterion — what "best" means for a lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Criterion {
    /// Lowest median latency.
    Fastest,
    /// Lowest max relative error vs the reference backend.
    MostAccurate,
    /// Weighted blend — lower is better. See module docs.
    Balanced,
}

impl Criterion {
    pub fn as_str(self) -> &'static str {
        match self {
            Criterion::Fastest      => "fastest",
            Criterion::MostAccurate => "accurate",
            Criterion::Balanced     => "balanced",
        }
    }
}

impl std::fmt::Display for Criterion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Weight applied to `max_rel_error` in the Balanced criterion's
/// cost function. Higher = more willing to give up speed for
/// accuracy. 100 ≈ "1% rel error is worth ~2× latency." Adjust via
/// [`DispatchTable::with_balanced_penalty`] if your workload is
/// unusually sensitive.
pub const DEFAULT_ACCURACY_PENALTY: f64 = 100.0;

/// Lookup key into [`DispatchTable`]. Combines the per-op axes
/// (what + on what data + how big) with the user's criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DispatchKey {
    op:         OpKind,
    dtype:      DType,
    size_class: SizeClass,
    criterion:  Criterion,
}

/// Where the dispatch table decided an op should run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pick {
    pub backend:      BackendId,
    pub device_index: u32,
}

/// O(1) runtime dispatch table, constructed once from a
/// [`ProfileReport`] and then queried at realize time.
#[derive(Debug, Clone)]
pub struct DispatchTable {
    entries: HashMap<DispatchKey, Pick>,
    /// All size classes present for each `(op, dtype)` — sorted
    /// ascending so `pick_nearest` can do a linear scan for the
    /// closest profiled bucket. Cheap because the table is built
    /// once and queried many times.
    size_index: HashMap<(OpKind, DType), Vec<SizeClass>>,
    accuracy_penalty: f64,
    include_reference: bool,
}

/// Options for building a [`DispatchTable`]. Use
/// [`DispatchOptions::default`] + the `with_*` setters.
#[derive(Debug, Clone, Copy)]
pub struct DispatchOptions {
    /// Include the reference backend in dispatch picks. Default
    /// false (reference is an oracle, not a production executor).
    pub include_reference: bool,
    /// Weight of `max_rel_error` in the Balanced criterion. Default
    /// [`DEFAULT_ACCURACY_PENALTY`].
    pub accuracy_penalty:  f64,
}

impl Default for DispatchOptions {
    fn default() -> Self {
        Self { include_reference: false, accuracy_penalty: DEFAULT_ACCURACY_PENALTY }
    }
}

impl DispatchOptions {
    pub fn with_reference_backend(mut self, include: bool) -> Self {
        self.include_reference = include;
        self
    }
    pub fn with_balanced_penalty(mut self, k: f64) -> Self {
        self.accuracy_penalty = k;
        self
    }
}

impl DispatchTable {
    /// Build a dispatch table from a profile report with default
    /// options. Reference backend excluded from picks; Balanced
    /// criterion uses [`DEFAULT_ACCURACY_PENALTY`].
    pub fn build(report: &ProfileReport) -> Self {
        Self::build_with(report, DispatchOptions::default())
    }

    /// Build with customised options — opt the reference backend
    /// back in, tune the Balanced penalty, etc.
    pub fn build_with(report: &ProfileReport, opts: DispatchOptions) -> Self {
        let mut tbl = Self {
            entries:           HashMap::new(),
            size_index:        HashMap::new(),
            accuracy_penalty:  opts.accuracy_penalty,
            include_reference: opts.include_reference,
        };
        tbl.rebuild_from(report);
        tbl
    }

    fn rebuild_from(&mut self, report: &ProfileReport) {
        self.entries.clear();
        self.size_index.clear();

        // Group entries by (op, dtype, size_class).
        let mut groups: HashMap<(OpKind, DType, SizeClass), Vec<&ProfileEntry>> = HashMap::new();
        for e in &report.entries {
            if !self.include_reference && e.backend == BackendId::Reference {
                continue;
            }
            groups.entry((e.op, e.dtype, e.size_class)).or_default().push(e);
        }

        for ((op, dtype, size_class), group) in &groups {
            for &criterion in &[Criterion::Fastest, Criterion::MostAccurate, Criterion::Balanced] {
                if let Some(winner) = self.pick_winner(group, criterion) {
                    let key = DispatchKey { op: *op, dtype: *dtype, size_class: *size_class, criterion };
                    self.entries.insert(key, Pick {
                        backend:      winner.backend,
                        device_index: winner.device_index,
                    });
                }
            }
            self.size_index.entry((*op, *dtype)).or_default().push(*size_class);
        }

        for classes in self.size_index.values_mut() {
            classes.sort_by_key(|s| s.0);
            classes.dedup();
        }
    }

    fn pick_winner<'a>(&self, group: &[&'a ProfileEntry], crit: Criterion) -> Option<&'a ProfileEntry> {
        match crit {
            Criterion::Fastest => group.iter().copied()
                .min_by_key(|e| e.latency_ns),
            Criterion::MostAccurate => group.iter().copied()
                .min_by(|a, b| {
                    a.max_rel_error.total_cmp(&b.max_rel_error)
                        .then(a.latency_ns.cmp(&b.latency_ns))
                }),
            Criterion::Balanced => group.iter().copied()
                .min_by(|a, b| {
                    let sa = a.latency_ns as f64 * (1.0 + self.accuracy_penalty * a.max_rel_error as f64);
                    let sb = b.latency_ns as f64 * (1.0 + self.accuracy_penalty * b.max_rel_error as f64);
                    sa.total_cmp(&sb)
                }),
        }
    }

    /// Exact lookup — returns `None` if the requested size class
    /// wasn't profiled. Use [`Self::pick_nearest`] for a nearest-
    /// neighbour fallback.
    pub fn pick(&self, op: OpKind, dtype: DType, size_class: SizeClass, criterion: Criterion) -> Option<Pick> {
        self.entries.get(&DispatchKey { op, dtype, size_class, criterion }).copied()
    }

    /// Nearest-neighbour lookup. If `size_class` wasn't measured
    /// exactly, pick the closest profiled class (ties go to the
    /// larger class, which is usually the safer scale-up).
    pub fn pick_nearest(&self, op: OpKind, dtype: DType, size_class: SizeClass, criterion: Criterion) -> Option<Pick> {
        if let Some(p) = self.pick(op, dtype, size_class, criterion) {
            return Some(p);
        }
        let classes = self.size_index.get(&(op, dtype))?;
        if classes.is_empty() {
            return None;
        }
        let target = size_class.0 as i32;
        let nearest = classes.iter()
            .min_by_key(|c| {
                let diff = (c.0 as i32 - target).abs();
                // Tie-break: prefer the larger size class. `(diff,
                // -c.0 as i32)` would work but we want `(diff,
                // -larger_bucket_number_in_unique_sort_key)`. Flip
                // the sign so smaller key = larger bucket wins at
                // tied diff.
                (diff, -(c.0 as i32))
            })?;
        self.pick(op, dtype, *nearest, criterion)
    }

    /// Total number of dispatch entries in the table. Useful for
    /// progress logging.
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Every distinct `(op, dtype, size_class)` for which the table
    /// has at least one criterion entry. Returns a sorted Vec (stable
    /// under repeat calls) — OpKind / DType / SizeClass don't
    /// implement Ord so the sort is by their string / u8 forms.
    pub fn keys(&self) -> Vec<(OpKind, DType, SizeClass)> {
        let mut seen: std::collections::HashSet<(OpKind, DType, SizeClass)> = Default::default();
        for k in self.entries.keys() {
            seen.insert((k.op, k.dtype, k.size_class));
        }
        let mut out: Vec<_> = seen.into_iter().collect();
        out.sort_by(|a, b| {
            a.0.as_str().cmp(b.0.as_str())
                .then(format!("{:?}", a.1).cmp(&format!("{:?}", b.1)))
                .then(a.2.0.cmp(&b.2.0))
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::{OpKind, ProfileEntry, ProfileReport, SizeClass, PROFILE_REPORT_VERSION};

    fn entry(backend: BackendId, op: OpKind, size: u8, latency: u64, err: f32) -> ProfileEntry {
        ProfileEntry {
            op,
            dtype:         DType::F32,
            size_class:    SizeClass(size),
            backend,
            device_index:  0,
            latency_ns:    latency,
            iterations:    7,
            max_rel_error: err,
        }
    }

    fn sample_report() -> ProfileReport {
        ProfileReport {
            version: PROFILE_REPORT_VERSION,
            entries: vec![
                // Size class 12: CUDA wins fastest (2ms < 10ms) but errs more
                entry(BackendId::Cpu,       OpKind::MatMul, 12, 10_000_000, 1e-6),
                entry(BackendId::Cuda,      OpKind::MatMul, 12,  2_000_000, 1e-4),
                entry(BackendId::Reference, OpKind::MatMul, 12,200_000_000, 0.0),
                // Size class 16: CPU is fastest + most accurate
                entry(BackendId::Cpu,       OpKind::MatMul, 16,   500_000, 1e-6),
                entry(BackendId::Cuda,      OpKind::MatMul, 16, 1_000_000, 1e-3),
                entry(BackendId::Reference, OpKind::MatMul, 16,15_000_000, 0.0),
            ],
        }
    }

    #[test]
    fn fastest_picks_lowest_latency_nonreference() {
        let tbl = DispatchTable::build(&sample_report());
        let p = tbl.pick(OpKind::MatMul, DType::F32, SizeClass(12), Criterion::Fastest).unwrap();
        assert_eq!(p, Pick { backend: BackendId::Cuda, device_index: 0 });
    }

    #[test]
    fn most_accurate_excludes_reference_by_default() {
        // Reference is EXCLUDED → CPU wins with 1e-6 over CUDA's 1e-4
        let tbl = DispatchTable::build(&sample_report());
        let p = tbl.pick(OpKind::MatMul, DType::F32, SizeClass(12), Criterion::MostAccurate).unwrap();
        assert_eq!(p, Pick { backend: BackendId::Cpu, device_index: 0 });
    }

    #[test]
    fn most_accurate_with_reference_opt_in() {
        let tbl = DispatchTable::build_with(
            &sample_report(),
            DispatchOptions::default().with_reference_backend(true),
        );
        let p = tbl.pick(OpKind::MatMul, DType::F32, SizeClass(12), Criterion::MostAccurate).unwrap();
        assert_eq!(p, Pick { backend: BackendId::Reference, device_index: 0 });
    }

    #[test]
    fn balanced_penalizes_numerically_sketchy_backends() {
        // At size class 16: CPU=500μs @ 1e-6, CUDA=1000μs @ 1e-3.
        // Balanced score: CPU ≈ 500_000 × (1 + 100*1e-6) = 500_050
        //                 CUDA ≈ 1_000_000 × (1 + 100*1e-3) = 1_100_000
        // CPU wins.
        let tbl = DispatchTable::build(&sample_report());
        let p = tbl.pick(OpKind::MatMul, DType::F32, SizeClass(16), Criterion::Balanced).unwrap();
        assert_eq!(p, Pick { backend: BackendId::Cpu, device_index: 0 });
    }

    #[test]
    fn pick_returns_none_for_unprofiled_class() {
        let tbl = DispatchTable::build(&sample_report());
        assert!(tbl.pick(OpKind::MatMul, DType::F32, SizeClass(20), Criterion::Fastest).is_none());
    }

    #[test]
    fn pick_nearest_falls_back_to_closest_profiled_class() {
        let tbl = DispatchTable::build(&sample_report());
        // 14 is equidistant from 12 and 16; tie goes to the larger.
        let p = tbl.pick_nearest(OpKind::MatMul, DType::F32, SizeClass(14), Criterion::Fastest).unwrap();
        // Size 16: CPU @ 500μs beats CUDA @ 1000μs.
        assert_eq!(p, Pick { backend: BackendId::Cpu, device_index: 0 });
        // Class 18 is closer to 16 than to 12 → pick reflects size 16.
        let p = tbl.pick_nearest(OpKind::MatMul, DType::F32, SizeClass(18), Criterion::Fastest).unwrap();
        assert_eq!(p, Pick { backend: BackendId::Cpu, device_index: 0 });
        // Class 8 is closer to 12 → pick reflects size 12 (CUDA fastest).
        let p = tbl.pick_nearest(OpKind::MatMul, DType::F32, SizeClass(8), Criterion::Fastest).unwrap();
        assert_eq!(p, Pick { backend: BackendId::Cuda, device_index: 0 });
    }

    #[test]
    fn pick_nearest_none_when_op_has_no_entries() {
        let tbl = DispatchTable::build(&sample_report());
        assert!(tbl.pick_nearest(OpKind::AddElementwise, DType::F32, SizeClass(10), Criterion::Fastest).is_none());
    }

    #[test]
    fn dispatch_table_keys_enumerate_all_profiled_classes() {
        let tbl = DispatchTable::build(&sample_report());
        let keys = tbl.keys();
        // Two size classes × one op × one dtype → 2 distinct profiled classes.
        assert_eq!(keys.len(), 2);
        assert!(keys.iter().any(|(op, _, s)| *op == OpKind::MatMul && s.0 == 12));
        assert!(keys.iter().any(|(op, _, s)| *op == OpKind::MatMul && s.0 == 16));
    }
}
