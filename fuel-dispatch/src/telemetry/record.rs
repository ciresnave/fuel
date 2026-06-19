//! The `DispatchRecord` / `MissRecord` JSONL wire schema.
//!
//! One compact JSON object per line (JSONL — append-friendly: a long run streams
//! without rewriting). The shapes mirror Baracuda's `DispatchRecord`/`MissRecord`
//! ask; `ImplId` and `StructureKeyToken` are the opaque join tokens (FKC identity
//! and Baracuda's structure key). `schema` versions the line so a v1 batch feed
//! and a future v2 live feed are distinguishable.

use serde::{Deserialize, Serialize};

use super::impl_id::ImplId;
use super::structure_key::StructureKeyToken;

/// The telemetry wire-format version stamped on every emitted record. Distinct
/// from the Judge's `PROFILE_REPORT_VERSION`.
pub const TELEMETRY_SCHEMA_VERSION: u32 = 1;

/// One emitted dispatch decision. Serialized as a single compact JSON line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DispatchRecord {
    /// Telemetry wire-format version ([`TELEMETRY_SCHEMA_VERSION`]).
    pub schema: u32,
    /// Baracuda's structure key for this dispatch site — an opaque token Fuel
    /// obtains by CALLING Baracuda's `structure_key` (never derives). `None`
    /// until the callable is linked (Coarse mode may emit without it).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structure_key: Option<StructureKeyToken>,
    /// The implementation that won this dispatch.
    pub chosen: ImplId,
    /// Every admitted alternative + its measured latency (Detailed mode). Empty
    /// in Coarse mode, and omitted from the wire when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<Candidate>,
    /// Aggregated hit count for this `(structure_key, chosen)` cell since the
    /// last flush.
    pub count: u64,
}

/// One admitted alternative + its empirical latency (the "loser" rows).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    /// The alternative's stable identity.
    pub impl_id: ImplId,
    /// Median nanoseconds from the Judge oracle; `None` = unmeasured cell (an
    /// oracle miss — never a fabricated `0`; the static estimate stood).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ns: Option<u64>,
}

/// A structural miss: the tightest admissible contract at this dispatch key was
/// a GENERIC one — a structure-specialized cell would have fit, but none is
/// registered. This is Baracuda's demand signal (`MissRecord.wanted`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissRecord {
    /// Telemetry wire-format version ([`TELEMETRY_SCHEMA_VERSION`]).
    pub schema: u32,
    /// The desired specialized cell — what structure-specialized kernel would
    /// have fit here (the structure key of the live operands).
    pub wanted: StructureKeyToken,
    /// The generic contract the planner actually fell back to.
    pub fallback: ImplId,
    /// Aggregated count of this miss since the last flush.
    pub count: u64,
    // `est_speedup` is deliberately OMITTED: it is inferable from the fallback's
    // own `DispatchRecord` (the retained loser timings), not estimated at miss
    // time. We drop the field rather than hold the dataset to compute it.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::{ImplId, StructureKeyToken};
    use fuel_core_types::dispatch::OpKind;
    use fuel_core_types::{BackendId, DType};

    fn baracuda_impl() -> ImplId {
        ImplId {
            backend: BackendId::Cuda,
            op: OpKind::MatMul,
            dtypes: vec![DType::F16, DType::F16, DType::F16],
            kernel_source: "baracuda".into(),
            kernel_revision_hash: 0x8f3c1a,
        }
    }

    /// A DispatchRecord serializes to exactly ONE compact JSONL line and
    /// round-trips through serde unchanged.
    #[test]
    fn dispatch_record_round_trips_as_one_jsonl_line() {
        let rec = DispatchRecord {
            schema: 1,
            structure_key: Some(StructureKeyToken("mm:innerdiv16:vec8:f16".into())),
            chosen: baracuda_impl(),
            candidates: vec![
                Candidate { impl_id: baracuda_impl(), latency_ns: Some(41_230) },
                Candidate {
                    impl_id: ImplId { kernel_source: "cublas".into(), ..baracuda_impl() },
                    latency_ns: Some(48_800),
                },
            ],
            count: 1024,
        };
        let line = serde_json::to_string(&rec).expect("serialize");
        assert!(!line.contains('\n'), "JSONL record must be a single line");
        let back: DispatchRecord = serde_json::from_str(&line).expect("deserialize");
        assert_eq!(rec, back);
    }

    /// `MissRecord` carries the demand signal and deliberately has NO
    /// `est_speedup` field (it is inferable from the fallback's DispatchRecord).
    #[test]
    fn miss_record_round_trips_and_has_no_est_speedup() {
        let miss = MissRecord {
            schema: 1,
            wanted: StructureKeyToken("mm:innerdiv16:vec8:flipped:f16".into()),
            fallback: ImplId { kernel_source: "baracuda-generic-strided".into(), ..baracuda_impl() },
            count: 37,
        };
        let line = serde_json::to_string(&miss).expect("serialize");
        assert!(!line.contains('\n'), "JSONL record must be a single line");
        assert!(!line.contains("est_speedup"), "est_speedup must be omitted");
        let back: MissRecord = serde_json::from_str(&line).expect("deserialize");
        assert_eq!(miss, back);
    }

    /// Coarse mode emits no candidates: an empty `candidates` vec is omitted
    /// from the wire form (so a Coarse record is minimal).
    #[test]
    fn empty_candidates_are_omitted_from_the_wire() {
        let rec = DispatchRecord {
            schema: 1,
            structure_key: None,
            chosen: baracuda_impl(),
            candidates: vec![],
            count: 5,
        };
        let line = serde_json::to_string(&rec).expect("serialize");
        assert!(!line.contains("candidates"), "empty candidates must be omitted");
        assert!(!line.contains("structure_key"), "None structure_key must be omitted");
    }
}
