//! The opt-in telemetry emission flag — **off by default**.
//!
//! No record is ever written unless emission is explicitly enabled. This is the
//! sole opt-in gate: a default-constructed [`TelemetryConfig`] is
//! [`TelemetryMode::Off`], and the sink is only ever touched when a caller both
//! flips the mode and threads the config in. There is **no env-var magic** and
//! nothing is on by default.

use std::path::PathBuf;

/// Telemetry emission mode. `Off` (the default) writes nothing and opens no
/// file; the coarse/detailed split governs whether per-candidate Judge timings
/// ride along.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TelemetryMode {
    /// Emission disabled — zero overhead, no record ever written (**DEFAULT**).
    #[default]
    Off,
    /// `(structure_key, chosen)` + aggregated counts and the miss histogram;
    /// **no** `candidates[]` (no Judge-oracle reads). Emits a `DispatchRecord`
    /// for every planned cell (the `chosen` signal) with an empty
    /// `candidates[]`, plus the miss histogram — which needs no Judge timings.
    Coarse,
    /// Coarse **plus** `candidates[]` with per-candidate Judge timings: each
    /// admitted alternative's own measured latency from the plan-time Judge
    /// oracle (`None` for unmeasured cells — never a fabricated `0`).
    Detailed,
}

impl TelemetryMode {
    /// Whether any emission happens at all in this mode.
    pub fn is_enabled(self) -> bool {
        !matches!(self, TelemetryMode::Off)
    }

    /// Whether `candidates[]` (per-candidate Judge timings) are populated.
    pub fn wants_candidates(self) -> bool {
        matches!(self, TelemetryMode::Detailed)
    }
}

/// The telemetry emission configuration. Default is `Off` with no path, so a
/// build that never sets it never emits.
#[derive(Debug, Clone, Default)]
pub struct TelemetryConfig {
    /// Emission mode (default [`TelemetryMode::Off`]).
    pub mode: TelemetryMode,
    /// Where the JSONL feed is flushed. `None` ⇒ the sink's caller supplies a
    /// path at flush time (the hardware-keyed `default_telemetry_path()`
    /// resolution is a `fuel-core` concern, landing with the dispatch-record
    /// half).
    pub out_path: Option<PathBuf>,
}

impl TelemetryConfig {
    /// Whether this config enables any emission.
    pub fn is_enabled(&self) -> bool {
        self.mode.is_enabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default is `Off` and emits nothing — the hard opt-in gate.
    #[test]
    fn default_mode_is_off_and_disabled() {
        let cfg = TelemetryConfig::default();
        assert_eq!(cfg.mode, TelemetryMode::Off);
        assert!(!cfg.is_enabled(), "default config must emit nothing");
        assert!(!TelemetryMode::default().is_enabled());
    }

    /// Coarse enables emission but not candidates; Detailed enables both.
    #[test]
    fn coarse_vs_detailed_candidates_gate() {
        assert!(TelemetryMode::Coarse.is_enabled());
        assert!(!TelemetryMode::Coarse.wants_candidates(), "coarse omits candidates");
        assert!(TelemetryMode::Detailed.is_enabled());
        assert!(TelemetryMode::Detailed.wants_candidates(), "detailed fills candidates");
    }
}
