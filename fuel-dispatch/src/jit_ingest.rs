//! Candidate-kernel ingestion (Spec B) — the seam types this module starts
//! from. This slice (Task 2) defines only the types a provider's rejection/
//! adoption feedback flows through: [`RejectionReport`], [`ProviderFeedback`],
//! and [`IngestOutcome`]. The probe/verify/adopt pipeline that produces these
//! values lands in later tasks.

use crate::fkc::verify::LedgerRecord;

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
