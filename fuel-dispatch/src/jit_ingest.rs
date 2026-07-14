//! Candidate-kernel ingestion (Spec B) — the seam types this module starts
//! from. Task 2 defined the types a provider's rejection/adoption feedback
//! flows through: [`RejectionReport`], [`ProviderFeedback`], and
//! [`IngestOutcome`]. Task 3 (this slice) adds [`CandidateKernel`] — the
//! not-yet-verified offer a provider hands Fuel, bundling the kernel fn
//! pointer with the operand/dtype shape facts and the precision claims it
//! *declares* (unverified until [`crate::fkc::verify`] empirically checks
//! them — see `jit_ingest_probe`'s `probe_from_operands` for the probe-input
//! synthesis step that feeds that verification). No consumer yet
//! (Task 5/6 wire ingest + verify around it) — `dead_code` is expected.

use crate::fkc::verify::LedgerRecord;

/// A not-yet-verified kernel a provider has offered Fuel: the callable
/// itself, the region it claims to replace (`decompose`, `None` if the
/// provider synthesized from scratch rather than fusing an existing
/// subgraph), the exact operand/dtype shapes it was built for, and the
/// precision guarantees it *declares* (unverified — [`crate::fkc::verify`]
/// is what turns a declared claim into an empirically-checked one before
/// `adopt_runtime_fused` ever sees it).
pub struct CandidateKernel {
    pub entry_point: String,
    pub kernel: crate::kernel::KernelRef,
    pub op_params: crate::kernel::OpParams,
    pub decompose: Option<fuel_graph::jit::PatternNode>,
    pub operands: Vec<baracuda_kernels_types::OperandDesc>,
    pub dtypes: Vec<fuel_ir::DType>,
    pub kernel_revision_hash: u64,
    pub declared: crate::fused::PrecisionGuarantee,
    pub backend: fuel_ir::probe::BackendId,
}

/// Why Fuel refused a candidate kernel a provider offered — handed to
/// [`ProviderFeedback::on_rejected`] so the provider can stop re-offering or
/// log the reason. `ledger_record` carries the empirical verification result
/// (if the rejection came from a failed [`crate::fkc::verify`] claim) rather
/// than a synthetic value.
pub struct RejectionReport {
    pub entry_point: String,
    pub failed_claim: &'static str,
    pub detail: String,
    pub ledger_record: Option<LedgerRecord>,
}

/// The callback surface a candidate-kernel provider implements to learn the
/// outcome of an offer. `on_rejected` is required (the whole point of the
/// report); `on_adopted` is optional telemetry, default no-op.
pub trait ProviderFeedback: Send + Sync {
    fn on_rejected(&self, report: &RejectionReport);
    fn on_adopted(&self, _entry_point: &str, _id: fuel_graph::registry::FusedOpId) {}
}

/// The result of ingesting one candidate kernel: adopted (with the
/// `FusedOpId` it registered under) or rejected (with the report explaining
/// why).
pub enum IngestOutcome {
    Adopted(fuel_graph::registry::FusedOpId),
    Rejected(RejectionReport),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_feedback_receives_the_report() {
        use std::sync::Mutex;
        struct Rec(Mutex<Vec<String>>);
        impl ProviderFeedback for Rec {
            fn on_rejected(&self, r: &RejectionReport) {
                self.0.lock().unwrap().push(r.failed_claim.into());
            }
        }
        let rec = Rec(Mutex::new(vec![]));
        rec.on_rejected(&RejectionReport {
            entry_point: "k".into(),
            failed_claim: "max_ulp",
            detail: "d".into(),
            ledger_record: None,
        });
        assert_eq!(rec.0.lock().unwrap().as_slice(), &["max_ulp".to_string()]);
    }
}
