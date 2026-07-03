//! The `DispatchRecord` / `MissRecord` JSONL wire schema.
//!
//! One compact JSON object per line (JSONL — append-friendly: a long run streams
//! without rewriting). The shapes mirror Baracuda's `DispatchRecord`/`MissRecord`
//! ask; `ImplId` and `StructureKeyToken` are the opaque join tokens (FKC identity
//! and Baracuda's structure key). `schema` versions the line: **v2** adds the
//! [`HwStamp`] hardware fingerprint to both records (v1 had none) so a merge can
//! arch-gate measurements and drop stampless rows rather than guess.

use serde::{Deserialize, Serialize};

use super::impl_id::ImplId;
use super::structure_key::StructureKeyToken;

/// The telemetry wire-format version stamped on every emitted record. Distinct
/// from the Judge's `PROFILE_REPORT_VERSION`.
///
/// **v2** (2026-07-03): both records carry a [`HwStamp`] hardware fingerprint
/// (`compute_capability` + `hardware_sku` + `driver_version`, mirroring the
/// device probe) so Baracuda's `merge` can arch-gate measurements. **v1** had no
/// stamp. Every wire-shape change bumps this so an old feed and a new feed stay
/// distinguishable.
pub const TELEMETRY_SCHEMA_VERSION: u32 = 2;

/// A hardware fingerprint stamped onto every emitted record (schema v2) so
/// Baracuda's `merge` can arch-gate measurements — rows from different silicon
/// or driver revisions never pool. Mirrors the device probe's field names +
/// types (`fuel-ir/src/probe.rs`, `DeviceDescriptor`).
///
/// `compute_capability` is `None` on non-CUDA backends; a record whose CC is
/// `None` is exactly the one a structure-key merge should DROP, never guess — so
/// "no CC ⇒ dropped, not fabricated" composes with the same posture as an
/// unmeasured `Candidate::latency_ns`. `None` is omitted from the wire.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HwStamp {
    /// CUDA compute capability `(major, minor)` — e.g. `(8, 9)` for sm_89.
    /// `None` on non-CUDA backends; omitted from the wire when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute_capability: Option<(u32, u32)>,
    /// Human-readable device name as the driver reports it (probe
    /// `hardware_sku`, e.g. `"NVIDIA GeForce RTX 4070"`).
    pub hardware_sku: String,
    /// Driver version string (probe `driver_version`) — an arch-gate axis the
    /// Judge's `EquivalenceKey` already splits on.
    pub driver_version: String,
}

impl HwStamp {
    /// Build a hardware fingerprint from the device probe descriptor
    /// (`fuel-ir` [`DeviceDescriptor`](fuel_ir::probe::DeviceDescriptor)) for
    /// the dispatching device. A CPU-only path yields
    /// `compute_capability: None` — fine: Baracuda's `merge` drops stampless
    /// CUDA rows, and CPU rows are retained for Fuel's own analysis. The field
    /// names + types mirror the probe exactly, so this is a pure projection.
    pub fn from_descriptor(desc: &fuel_ir::probe::DeviceDescriptor) -> Self {
        Self {
            compute_capability: desc.compute_capability,
            hardware_sku: desc.hardware_sku.clone(),
            driver_version: desc.driver_version.clone(),
        }
    }
}

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
    /// Hardware fingerprint of the device this dispatch ran on (schema v2).
    /// Lets Baracuda's `merge` arch-gate the row; a record whose
    /// `hw.compute_capability` is `None` (non-CUDA) is one the merge drops.
    pub hw: HwStamp,
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
    /// Hardware fingerprint of the device (schema v2; see [`DispatchRecord::hw`]).
    pub hw: HwStamp,
    // `est_speedup` is deliberately OMITTED: it is inferable from the fallback's
    // own `DispatchRecord` (the retained loser timings), not estimated at miss
    // time. We drop the field rather than hold the dataset to compute it.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::{ImplId, StructureKeyToken};
    use fuel_ir::dispatch::OpKind;
    use fuel_ir::{BackendId, DType};

    fn baracuda_impl() -> ImplId {
        ImplId {
            backend: BackendId::Cuda,
            op: OpKind::MatMul,
            dtypes: vec![DType::F16, DType::F16, DType::F16],
            kernel_source: "baracuda".into(),
            kernel_revision_hash: 0x8f3c1a,
        }
    }

    /// A CUDA hardware fingerprint (compute_capability present).
    fn cuda_stamp() -> HwStamp {
        HwStamp {
            compute_capability: Some((8, 9)),
            hardware_sku: "NVIDIA GeForce RTX 4070".into(),
            driver_version: "552.44".into(),
        }
    }

    /// A non-CUDA (CPU) fingerprint: `compute_capability: None`.
    fn cpu_stamp() -> HwStamp {
        HwStamp {
            compute_capability: None,
            hardware_sku: "Intel(R) Core(TM) i9-14900K".into(),
            driver_version: "n/a".into(),
        }
    }

    /// A DispatchRecord serializes to exactly ONE compact JSONL line and
    /// round-trips through serde unchanged.
    #[test]
    fn dispatch_record_round_trips_as_one_jsonl_line() {
        let rec = DispatchRecord {
            schema: TELEMETRY_SCHEMA_VERSION,
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
            hw: cuda_stamp(),
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
            schema: TELEMETRY_SCHEMA_VERSION,
            wanted: StructureKeyToken("mm:innerdiv16:vec8:flipped:f16".into()),
            fallback: ImplId { kernel_source: "baracuda-generic-strided".into(), ..baracuda_impl() },
            count: 37,
            hw: cuda_stamp(),
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
            schema: TELEMETRY_SCHEMA_VERSION,
            structure_key: None,
            chosen: baracuda_impl(),
            candidates: vec![],
            count: 5,
            hw: cuda_stamp(),
        };
        let line = serde_json::to_string(&rec).expect("serialize");
        assert!(!line.contains("candidates"), "empty candidates must be omitted");
        assert!(!line.contains("structure_key"), "None structure_key must be omitted");
    }

    /// The schema-v2 hardware fingerprint serializes on both records, and the
    /// non-CUDA (`compute_capability: None`) case is handled: a CUDA stamp
    /// carries its `(major, minor)` on the wire; a CPU stamp OMITS
    /// `compute_capability` yet round-trips back to `None` (the
    /// "stampless-CC ⇒ dropped, not guessed" case Baracuda's merge relies on).
    #[test]
    fn hw_stamp_serializes_and_handles_non_cuda_none() {
        // CUDA stamp: compute_capability present on the wire + round-trips.
        let rec = DispatchRecord {
            schema: TELEMETRY_SCHEMA_VERSION,
            structure_key: None,
            chosen: baracuda_impl(),
            candidates: vec![],
            count: 1,
            hw: cuda_stamp(),
        };
        let line = serde_json::to_string(&rec).expect("serialize");
        assert!(line.contains("compute_capability"), "CUDA stamp carries CC");
        assert!(line.contains("hardware_sku"), "sku present");
        assert!(line.contains("driver_version"), "driver present");
        let back: DispatchRecord = serde_json::from_str(&line).expect("deserialize");
        assert_eq!(rec, back);
        assert_eq!(back.hw.compute_capability, Some((8, 9)));

        // Non-CUDA stamp: compute_capability is None ⇒ omitted from the wire,
        // still round-trips to None (sku + driver remain).
        let miss = MissRecord {
            schema: TELEMETRY_SCHEMA_VERSION,
            wanted: StructureKeyToken("mm:innerdiv16:vec8:f16".into()),
            fallback: baracuda_impl(),
            count: 3,
            hw: cpu_stamp(),
        };
        let line = serde_json::to_string(&miss).expect("serialize");
        assert!(!line.contains("compute_capability"), "None CC must be omitted from the wire");
        assert!(line.contains("hardware_sku") && line.contains("driver_version"), "sku/driver stay");
        let back: MissRecord = serde_json::from_str(&line).expect("deserialize");
        assert_eq!(miss, back);
        assert_eq!(back.hw.compute_capability, None);
    }

    /// The schema number is bumped to v2 for the hardware-fingerprint addition.
    #[test]
    fn telemetry_schema_version_is_v2_for_hw_stamp() {
        assert_eq!(TELEMETRY_SCHEMA_VERSION, 2);
    }

    /// `HwStamp::from_descriptor` is a pure projection of the device probe.
    /// A CUDA descriptor carries its compute capability; a CPU-only
    /// descriptor yields `compute_capability: None` (the stampless-CUDA-row
    /// case Baracuda's merge drops, CPU rows kept for our own analysis).
    #[test]
    fn hw_stamp_from_descriptor_projects_probe_fields() {
        use fuel_ir::probe::DeviceDescriptor;
        use fuel_ir::{BackendId, DeviceLocation};

        let cuda = DeviceDescriptor {
            backend: BackendId::Cuda,
            device_index: 0,
            hardware_sku: "NVIDIA GeForce RTX 4070".into(),
            vendor_id: 0x10DE,
            device_id: 0x2786,
            compute_capability: Some((8, 9)),
            driver_version: "552.44".into(),
            total_memory_bytes: 12 * 1024 * 1024 * 1024,
            location: DeviceLocation::Cuda { gpu_id: 0 },
        };
        let stamp = HwStamp::from_descriptor(&cuda);
        assert_eq!(stamp.compute_capability, Some((8, 9)));
        assert_eq!(stamp.hardware_sku, "NVIDIA GeForce RTX 4070");
        assert_eq!(stamp.driver_version, "552.44");

        let cpu = DeviceDescriptor {
            backend: BackendId::Cpu,
            device_index: 0,
            hardware_sku: "Intel(R) Core(TM) i9-14900K".into(),
            vendor_id: 0,
            device_id: 0,
            compute_capability: None,
            driver_version: "n/a".into(),
            total_memory_bytes: 0,
            location: DeviceLocation::Cpu,
        };
        let stamp = HwStamp::from_descriptor(&cpu);
        assert_eq!(stamp.compute_capability, None, "CPU path ⇒ no compute capability");
        assert_eq!(stamp.hardware_sku, "Intel(R) Core(TM) i9-14900K");
    }
}
