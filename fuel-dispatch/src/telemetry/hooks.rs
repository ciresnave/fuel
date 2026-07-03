//! The plan-time telemetry carrier — what `compile_plan` needs in hand to
//! emit a structural miss at the dispatch pick site.
//!
//! [`crate::plan::PlanOptions`] threads an `Option<&TelemetryHooks>`; when it
//! is `None` (the default) OR the config's mode is
//! [`TelemetryMode::Off`](super::config::TelemetryMode::Off), the plan's
//! emission post-pass returns before touching anything — the opt-in gate is
//! branch-predictable and the default build path is untouched (the whole
//! module is behind the `telemetry` cargo feature besides).
//!
//! ## Never-panic posture
//!
//! The `sink` is shared behind a [`Mutex`] because the plan's emission is a
//! `&`-borrow of the options while [`TelemetrySink::record_miss`] needs `&mut`.
//! `record_miss` cannot fail (pure in-memory aggregation); the only failure
//! mode at the call site is a poisoned lock, which the emission swallows to a
//! no-op — telemetry is best-effort and must never break dispatch.

use std::sync::Mutex;

use super::config::TelemetryConfig;
use super::record::HwStamp;
use super::sink::TelemetrySink;
use super::structure_key::StructureKeyProvider;

/// Everything the plan-time miss emission needs, threaded through
/// [`crate::plan::PlanOptions::telemetry`]. Built once per realize/compile
/// call and borrowed by [`crate::plan::compile_plan`].
pub struct TelemetryHooks<'env> {
    /// The opt-in emission configuration. `Off` (the default) makes the
    /// emission post-pass a no-op; `Coarse`/`Detailed` both emit misses (the
    /// miss half needs no Judge data, so Coarse is sufficient).
    pub config: &'env TelemetryConfig,
    /// The aggregating sink, shared behind a `Mutex` (see module docs). Every
    /// detected miss is folded into its `(wanted, fallback, hw)` cell.
    pub sink: &'env Mutex<TelemetrySink>,
    /// Baracuda's structure-key callable (the seam Fuel CALLS, never derives).
    /// The v1 default is the unlinked
    /// [`NullStructureKeyProvider`](super::structure_key::NullStructureKeyProvider),
    /// which yields no token ⇒ no demand signal (honest "unlinked" posture).
    pub provider: &'env dyn StructureKeyProvider,
    /// The hardware fingerprint of the dispatching device — stamped onto every
    /// emitted record so Baracuda's `merge` can arch-gate. A CPU-only realize
    /// carries `compute_capability: None`, which is exactly the "stampless-CC
    /// ⇒ dropped, not guessed" case the merge relies on.
    pub hw: HwStamp,
}

impl TelemetryHooks<'_> {
    /// The architecture tag handed to the structure-key provider (Baracuda
    /// keys its matrix per arch). Derived from the hardware fingerprint:
    /// `sm_<major><minor>` for a CUDA device, `"cpu"` when there is no compute
    /// capability. Baracuda owns the tag's meaning; Fuel supplies a stable
    /// string.
    pub fn arch_tag(&self) -> String {
        match self.hw.compute_capability {
            Some((major, minor)) => format!("sm_{major}{minor}"),
            None => "cpu".to_string(),
        }
    }
}
