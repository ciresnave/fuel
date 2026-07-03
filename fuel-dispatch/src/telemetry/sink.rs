//! The batch/offline JSONL sink — in-memory aggregation, flushed at run cadence.
//!
//! A process accumulates aggregated [`MissRecord`] counts in memory (keyed by
//! `(wanted, fallback, hw)`), then flushes them to a JSONL artifact (one compact
//! JSON object per line) via an atomic tmp+rename at process end / explicit
//! [`TelemetrySink::flush`]. This mirrors how the Judge's `ProfileReport` is
//! written once, but append-friendly JSONL so a long run streams without
//! rewriting.
//!
//! # Crate placement (divergence from the plan's step 7, justified)
//!
//! The plan puts the sink in `fuel-core` because it needs the concrete
//! `ProfileJudgeOracle` (to fill `candidates[]`) + the hardware-keyed cache dir
//! (`default_report_path`). **The MISS half needs neither** — it aggregates
//! records and writes JSONL to a caller-supplied path, with **no Judge read**.
//! So the miss-first sink lives in `fuel-dispatch`, alongside the record types
//! it aggregates: the whole miss slice stays in one crate (one cargo
//! invocation, no premature `fuel-core` oracle machinery). When the
//! dispatch-record / `candidates[]` half lands, its Detailed-mode fill (which
//! *does* need the oracle) is what pulls a writer into `fuel-core`; the
//! aggregation shape here is a strict subset that half extends.
//!
//! # Never-panic posture
//!
//! Emission must never break dispatch. [`TelemetrySink::record_miss`] cannot
//! fail (pure in-memory aggregation). [`TelemetrySink::flush`] returns
//! `io::Result` so a caller logs a write failure to telemetry rather than
//! letting it propagate into the dispatch path — the sink never `panic!`s.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::impl_id::ImplId;
use super::record::{HwStamp, MissRecord};
use super::structure_key::StructureKeyToken;

/// The aggregation key for a miss cell: `(wanted, fallback, hw)`. The hardware
/// stamp is part of the identity so rows from different silicon / driver
/// revisions never pool (mirroring Baracuda's merge arch-gate); a single-device
/// run collapses to one `hw` and aggregates purely on `(wanted, fallback)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MissKey {
    wanted: StructureKeyToken,
    fallback: ImplId,
    hw: HwStamp,
}

/// In-memory aggregation of emitted telemetry, flushed to JSONL at run/release
/// cadence (batch/offline v1). Miss records aggregate by `(wanted, fallback,
/// hw)`, so a long run collapses to a histogram rather than one line per
/// dispatch.
///
/// The dispatch-record half (`DispatchRecord` aggregation + `candidates[]`
/// fill) is **not** built here yet — see the module docs.
#[derive(Debug, Default)]
pub struct TelemetrySink {
    /// Aggregated miss observations: cell → summed count.
    misses: HashMap<MissKey, u64>,
}

impl TelemetrySink {
    /// A fresh, empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one observed structural miss, aggregating into the
    /// `(wanted, fallback, hw)` cell. A record's own `count` (≥ 1) is added, so
    /// pre-aggregated records merge correctly too. **Never fails** — pure
    /// in-memory aggregation, so emission can never break dispatch.
    pub fn record_miss(&mut self, miss: MissRecord) {
        let key = MissKey {
            wanted: miss.wanted,
            fallback: miss.fallback,
            hw: miss.hw,
        };
        *self.misses.entry(key).or_insert(0) += miss.count.max(1);
    }

    /// Number of distinct aggregated miss cells (observability / test helper).
    pub fn miss_cell_count(&self) -> usize {
        self.misses.len()
    }

    /// Materialize the aggregated misses as [`MissRecord`]s — one per cell,
    /// `count` = summed observations. Sorted for a deterministic, diff-friendly
    /// feed (by `wanted`, then the fallback's source + revision).
    pub fn miss_records(&self) -> Vec<MissRecord> {
        let mut recs: Vec<MissRecord> = self
            .misses
            .iter()
            .map(|(key, &count)| MissRecord {
                schema: super::record::TELEMETRY_SCHEMA_VERSION,
                wanted: key.wanted.clone(),
                fallback: key.fallback.clone(),
                count,
                hw: key.hw.clone(),
            })
            .collect();
        recs.sort_by(|a, b| {
            a.wanted
                .0
                .cmp(&b.wanted.0)
                .then_with(|| a.fallback.kernel_source.cmp(&b.fallback.kernel_source))
                .then_with(|| a.fallback.kernel_revision_hash.cmp(&b.fallback.kernel_revision_hash))
        });
        recs
    }

    /// Whether the sink holds anything to flush.
    pub fn is_empty(&self) -> bool {
        self.misses.is_empty()
    }

    /// Drain the aggregated records to a JSONL file — one compact JSON object
    /// per line — via an atomic tmp+rename. Returns the number of lines
    /// written. The `io::Result` surfaces a write failure to the caller (a
    /// telemetry log); it never propagates into dispatch.
    pub fn flush(&self, path: &Path) -> std::io::Result<usize> {
        let records = self.miss_records();
        let mut body = String::new();
        for r in &records {
            // These record types are plain data; serialization cannot fail,
            // but map the error rather than unwrap to keep the path panic-free.
            let line = serde_json::to_string(r).map_err(std::io::Error::other)?;
            body.push_str(&line);
            body.push('\n');
        }
        // Atomic write: stage to a sibling tmp file, then rename over the
        // target (atomic on the same directory / filesystem).
        let mut tmp: PathBuf = path.to_path_buf();
        tmp.as_mut_os_string().push(".tmp");
        std::fs::write(&tmp, body.as_bytes())?;
        std::fs::rename(&tmp, path)?;
        Ok(records.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::miss::{detect_miss, AdmittedContract};
    use crate::telemetry::structure_key::{Contiguity, FdxOperandDesc, StructureKeyProvider};
    use crate::fkc::{ResolvedLayout, Tri};
    use fuel_ir::dispatch::OpKind;
    use fuel_ir::{BackendId, DType};

    struct CannedProvider(String);
    impl StructureKeyProvider for CannedProvider {
        fn structure_key(
            &self,
            op_class: &str,
            _operands: &[FdxOperandDesc],
            _arch: &str,
        ) -> Option<StructureKeyToken> {
            Some(StructureKeyToken(format!("{op_class}:{}", self.0)))
        }
    }

    fn impl_id(kernel_source: &str) -> ImplId {
        ImplId {
            backend: BackendId::Cuda,
            op: OpKind::MatMul,
            dtypes: vec![DType::F16, DType::F16, DType::F16],
            kernel_source: kernel_source.into(),
            kernel_revision_hash: 0xabc,
        }
    }

    /// A CPU fingerprint (`compute_capability: None`) — exercises the stampless
    /// wire path through the sink.
    fn cpu_hw() -> HwStamp {
        HwStamp {
            compute_capability: None,
            hardware_sku: "Intel(R) Core(TM) i9-14900K".into(),
            driver_version: "n/a".into(),
        }
    }

    fn generic_best() -> AdmittedContract {
        AdmittedContract {
            impl_id: impl_id("baracuda-generic-strided"),
            layouts: vec![
                ResolvedLayout {
                    contiguous: Tri::NotApplicable,
                    strided: Tri::Accepted,
                    broadcast_stride0: Tri::Accepted,
                    start_offset: Tri::Rejected,
                    reverse_strides: Tri::Rejected,
                };
                2
            ],
        }
    }

    fn operand() -> FdxOperandDesc {
        FdxOperandDesc {
            dtype: DType::F16,
            contiguity: Contiguity::Contiguous,
            broadcast: false,
            flipped: false,
        }
    }

    /// BORN-RED (the headline): three dispatches at a generic-only cell emit
    /// exactly ONE aggregated `MissRecord` with the canned token, the correct
    /// fallback `ImplId`, and `count == 3`.
    #[test]
    fn generic_only_cell_emits_one_aggregated_miss_record() {
        let provider = CannedProvider("innerdiv16:vec8:f16".into());
        let best = generic_best();
        let mut sink = TelemetrySink::new();
        for _ in 0..3 {
            let miss = detect_miss(&best, "matmul", &[operand()], "sm_89", &provider, cpu_hw())
                .expect("generic-only cell must emit a miss");
            sink.record_miss(miss);
        }
        assert_eq!(sink.miss_cell_count(), 1, "three identical misses aggregate to ONE cell");
        let recs = sink.miss_records();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].wanted, StructureKeyToken("matmul:innerdiv16:vec8:f16".into()));
        assert_eq!(recs[0].fallback, impl_id("baracuda-generic-strided"));
        assert_eq!(recs[0].count, 3, "the three observations sum in the cell count");
        assert_eq!(recs[0].hw, cpu_hw());
    }

    /// Distinct fallbacks stay separate cells (aggregation keys on the tuple).
    #[test]
    fn distinct_fallbacks_are_distinct_cells() {
        let provider = CannedProvider("k".into());
        let mut sink = TelemetrySink::new();
        for src in ["baracuda-generic-strided", "portable-cpu-strided"] {
            let best = AdmittedContract { impl_id: impl_id(src), ..generic_best() };
            let miss = detect_miss(&best, "matmul", &[operand()], "sm_89", &provider, cpu_hw())
                .expect("miss");
            sink.record_miss(miss);
        }
        assert_eq!(sink.miss_cell_count(), 2, "two fallbacks ⇒ two cells");
    }

    /// The sink flushes valid JSONL (each line parses standalone as a
    /// `MissRecord`) and round-trips against the pinned v2 schema — including
    /// the `HwStamp` `compute_capability: None` (CPU) case.
    #[test]
    fn flush_writes_valid_jsonl_that_round_trips() {
        let provider = CannedProvider("innerdiv16:f16".into());
        let best = generic_best();
        let mut sink = TelemetrySink::new();
        let miss = detect_miss(&best, "matmul", &[operand()], "sm_89", &provider, cpu_hw())
            .expect("miss");
        sink.record_miss(miss);

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("telemetry.jsonl");
        let n = sink.flush(&path).expect("flush");
        assert_eq!(n, 1, "one aggregated miss ⇒ one line");

        let body = std::fs::read_to_string(&path).expect("read back");
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 1, "one record ⇒ one JSONL line");
        // Each line parses standalone as a MissRecord and round-trips.
        let back: MissRecord = serde_json::from_str(lines[0]).expect("parse MissRecord line");
        assert_eq!(back.wanted, StructureKeyToken("matmul:innerdiv16:f16".into()));
        assert_eq!(back.fallback, impl_id("baracuda-generic-strided"));
        assert_eq!(back.hw.compute_capability, None, "CPU stamp round-trips to None");
        // v2 schema stamped; est_speedup deliberately absent from the wire.
        assert_eq!(back.schema, super::super::record::TELEMETRY_SCHEMA_VERSION);
        assert!(!lines[0].contains("est_speedup"), "est_speedup is omitted");
        assert!(!lines[0].contains("compute_capability"), "None CC omitted from the wire");
    }

    /// An empty sink flushes an empty file without error (never-panic).
    #[test]
    fn empty_sink_flushes_empty_file() {
        let sink = TelemetrySink::new();
        assert!(sink.is_empty());
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("empty.jsonl");
        let n = sink.flush(&path).expect("flush empty");
        assert_eq!(n, 0);
        assert_eq!(std::fs::read_to_string(&path).expect("read").len(), 0);
    }
}
