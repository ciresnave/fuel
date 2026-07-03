//! Baracuda dispatch-telemetry / miss-reporting — the **production consumer**.
//!
//! `fuel-dispatch` owns the record types + the plan-time emission (behind its
//! own `telemetry` feature, which this feature enables). `fuel-core` owns the
//! *process-wide* pieces that a real realize needs: the opt-in switch, the
//! aggregating sink, the hardware fingerprint, the on-disk output path, and the
//! explicit flush. The realize path (`pipelined_bridge::build_optimized_graph`)
//! reads this module's opt-in state and installs the plan-time
//! [`fuel_dispatch::telemetry::TelemetryHooks`] when enabled.
//!
//! # Opt-in, and no automatic IO
//!
//! Off by default: nothing is emitted until [`enable`] is called, and nothing
//! is ever written to disk except by an explicit [`flush_to`] / [`flush`] call.
//! There is no env-var magic and no background writer thread — a flush happens
//! only when the caller asks for one.
//!
//! # v1 provider = Null (honest "unlinked")
//!
//! Baracuda's `structure_key` callable is cuda-gated FFI and not linked in this
//! environment, so the installed provider is the
//! [`fuel_dispatch::telemetry::NullStructureKeyProvider`]: dispatch records are
//! emitted without a structure key and no miss demand signal forms (never a
//! fabricated token). When Baracuda ships the callable, only the provider swaps.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

use fuel_dispatch::telemetry::{
    HwStamp, NullStructureKeyProvider, TelemetryConfig, TelemetryHooks, TelemetryMode,
    TelemetrySink,
};
use fuel_ir::DeviceLocation;

const MODE_OFF: u8 = 0;
const MODE_COARSE: u8 = 1;
const MODE_DETAILED: u8 = 2;

fn mode_to_u8(m: TelemetryMode) -> u8 {
    match m {
        TelemetryMode::Off => MODE_OFF,
        TelemetryMode::Coarse => MODE_COARSE,
        TelemetryMode::Detailed => MODE_DETAILED,
    }
}

fn u8_to_mode(v: u8) -> TelemetryMode {
    match v {
        MODE_COARSE => TelemetryMode::Coarse,
        MODE_DETAILED => TelemetryMode::Detailed,
        _ => TelemetryMode::Off,
    }
}

/// Process-wide telemetry state: the opt-in mode (an atomic so
/// [`enable`]/[`disable`] are lock-free) + the aggregating sink.
struct GlobalTelemetry {
    mode: AtomicU8,
    sink: Mutex<TelemetrySink>,
}

static TELEMETRY: OnceLock<GlobalTelemetry> = OnceLock::new();

fn global() -> &'static GlobalTelemetry {
    TELEMETRY.get_or_init(|| GlobalTelemetry {
        mode: AtomicU8::new(MODE_OFF),
        sink: Mutex::new(TelemetrySink::new()),
    })
}

/// Turn emission on at `mode` ([`TelemetryMode::Coarse`] or
/// [`TelemetryMode::Detailed`]). Idempotent; changes take effect on the next
/// realize. `Off` is equivalent to [`disable`].
pub fn enable(mode: TelemetryMode) {
    global().mode.store(mode_to_u8(mode), Ordering::Relaxed);
}

/// Turn emission off. Accumulated records are retained until [`reset`] or a
/// [`flush_to`]; a subsequent [`enable`] resumes into the same sink.
pub fn disable() {
    global().mode.store(MODE_OFF, Ordering::Relaxed);
}

/// The current emission mode (default [`TelemetryMode::Off`]).
pub fn current_mode() -> TelemetryMode {
    u8_to_mode(global().mode.load(Ordering::Relaxed))
}

/// Whether emission is currently enabled.
pub fn is_enabled() -> bool {
    current_mode().is_enabled()
}

/// The process-wide sink the plan-time hook records into. Internal — the realize
/// path threads it into [`TelemetryHooks`], and [`flush_to`] drains it.
pub(crate) fn sink() -> &'static Mutex<TelemetrySink> {
    &global().sink
}

/// Discard all accumulated records (leaves the mode unchanged). Use between
/// runs whose feeds should not pool.
pub fn reset() {
    if let Ok(mut s) = global().sink.lock() {
        *s = TelemetrySink::new();
    }
}

/// Explicitly flush the accumulated feed into `dir` as the two homogeneous
/// files `misses.jsonl` + `dispatches.jsonl`. Returns `(miss_lines,
/// dispatch_lines)`. Creates `dir` if needed. The only path that writes to
/// disk — never automatic.
pub fn flush_to(dir: &Path) -> std::io::Result<(usize, usize)> {
    let s = global()
        .sink
        .lock()
        .map_err(|_| std::io::Error::other("telemetry sink lock poisoned"))?;
    s.flush_all(dir)
}

/// Flush to the default hardware-keyed telemetry directory
/// ([`default_telemetry_dir`]). `None` when no cache directory resolves (then
/// the caller must pick a path and use [`flush_to`]).
pub fn flush() -> Option<std::io::Result<(usize, usize)>> {
    default_telemetry_dir().map(|d| flush_to(&d))
}

/// The default telemetry output directory: the same hardware-keyed cache
/// directory that holds the Judge profile report, so the dispatch/miss feed
/// lands beside the profile it was measured against. `None` when no cache dir
/// is resolvable (headless / restricted environments).
pub fn default_telemetry_dir() -> Option<PathBuf> {
    crate::judge::default_report_path().and_then(|p| p.parent().map(|d| d.to_path_buf()))
}

/// Per-device hardware fingerprints, probed ONCE and cached (the probe
/// enumerates every backend's devices, so we pay it a single time per process).
static HW_STAMPS: OnceLock<Vec<(DeviceLocation, HwStamp)>> = OnceLock::new();

fn hw_stamps() -> &'static Vec<(DeviceLocation, HwStamp)> {
    HW_STAMPS.get_or_init(|| {
        fuel_hardware::probe::ProbeReport::probe_all()
            .devices
            .iter()
            .map(|d| (d.location, HwStamp::from_descriptor(d)))
            .collect()
    })
}

/// The hardware fingerprint for `device` (the realize's pinned device),
/// stamped onto every emitted record so Baracuda's `merge` can arch-gate. Falls
/// back to a `compute_capability: None` stamp when the device isn't in the
/// probe (a CPU-only realize on a headless box) — the honest "no CUDA silicon"
/// case the merge drops.
pub(crate) fn hw_stamp_for(device: DeviceLocation) -> HwStamp {
    hw_stamps()
        .iter()
        .find(|(loc, _)| *loc == device)
        .map(|(_, s)| s.clone())
        .unwrap_or_else(|| HwStamp {
            compute_capability: None,
            hardware_sku: format!("{device:?}"),
            driver_version: "unknown".into(),
        })
}

/// Owns the plan-time telemetry hook inputs for one realize (the config, the
/// Null provider, and the device stamp) so a [`TelemetryHooks`] borrowing them
/// can be threaded into `PlanOptions` with a lifetime that outlives the plan.
///
/// Constructed once per `build_optimized_graph` call; [`Self::hooks`] returns
/// `None` when emission is off (the realize path then threads no hooks and the
/// plan is byte-identical).
pub struct TelemetryInstall {
    config: TelemetryConfig,
    provider: NullStructureKeyProvider,
    hw: HwStamp,
    enabled: bool,
}

impl TelemetryInstall {
    /// Snapshot the opt-in state for a realize pinned to `device`.
    pub fn new(device: DeviceLocation) -> Self {
        let mode = current_mode();
        Self {
            config: TelemetryConfig { mode, out_path: None },
            provider: NullStructureKeyProvider,
            hw: hw_stamp_for(device),
            enabled: mode.is_enabled(),
        }
    }

    /// The plan-time hooks to thread into `PlanOptions::with_telemetry`, or
    /// `None` when emission is off. The hooks borrow `self` (config + provider)
    /// and the process-wide [`sink`]; keep `self` alive until after the plan.
    pub fn hooks(&self) -> Option<TelemetryHooks<'_>> {
        if !self.enabled {
            return None;
        }
        Some(TelemetryHooks {
            config: &self.config,
            sink: sink(),
            provider: &self.provider,
            hw: self.hw.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_dispatch::telemetry::{
        Candidate, DispatchRecord, HwStamp, ImplId, MissRecord, StructureKeyToken,
        TELEMETRY_SCHEMA_VERSION,
    };
    use fuel_ir::dispatch::OpKind;
    use fuel_ir::{BackendId, DType};

    fn impl_id(src: &str) -> ImplId {
        ImplId {
            backend: BackendId::Cpu,
            op: OpKind::AddElementwise,
            dtypes: vec![DType::F32, DType::F32, DType::F32],
            kernel_source: src.into(),
            kernel_revision_hash: 0,
        }
    }

    fn cpu_hw() -> HwStamp {
        HwStamp {
            compute_capability: None,
            hardware_sku: "test-cpu".into(),
            driver_version: "n/a".into(),
        }
    }

    /// The opt-in mode round-trip + the explicit flush, in ONE test.
    ///
    /// The process-wide sink is shared with any concurrently-running realize
    /// test, so this test is deliberately **race-robust**: it never asserts the
    /// sink is empty or an exact line count. It (a) round-trips the mode atomic
    /// (no sink interaction), then (b) records a UNIQUELY-keyed dispatch + miss
    /// directly into the sink (recording is independent of the mode, so this
    /// opens no emission window) and flushes, asserting OUR records are PRESENT
    /// among possibly-more lines.
    #[test]
    fn opt_in_state_machine_and_explicit_flush() {
        // (a) Mode atomic round-trip — no sink interaction.
        enable(TelemetryMode::Coarse);
        assert_eq!(current_mode(), TelemetryMode::Coarse);
        assert!(is_enabled());
        enable(TelemetryMode::Detailed);
        assert_eq!(current_mode(), TelemetryMode::Detailed);
        disable();
        assert_eq!(current_mode(), TelemetryMode::Off);
        assert!(!is_enabled());

        // (b) A structure key unique to this test run so a concurrent realize's
        // records (Null provider ⇒ `None` key) can never collide with ours.
        let uniq = format!("mm:telemetry-test:{:?}", std::thread::current().id());
        {
            let mut s = sink().lock().expect("sink");
            s.record_dispatch(DispatchRecord {
                schema: TELEMETRY_SCHEMA_VERSION,
                structure_key: Some(StructureKeyToken(uniq.clone())),
                chosen: impl_id("portable-cpu"),
                candidates: vec![Candidate {
                    impl_id: impl_id("portable-cpu"),
                    latency_ns: Some(41_230),
                }],
                count: 1,
                hw: cpu_hw(),
            });
            s.record_miss(MissRecord {
                schema: TELEMETRY_SCHEMA_VERSION,
                wanted: StructureKeyToken(uniq.clone()),
                fallback: impl_id("baracuda-generic-strided"),
                count: 1,
                hw: cpu_hw(),
            });
        }

        // Explicit flush → the two separate homogeneous JSONL files.
        let dir = tempfile::tempdir().expect("tempdir");
        let (misses, dispatches) = flush_to(dir.path()).expect("flush_to");
        assert!(misses >= 1 && dispatches >= 1, "at least our two lines");

        let disp = std::fs::read_to_string(dir.path().join("dispatches.jsonl")).expect("dispatches");
        let miss = std::fs::read_to_string(dir.path().join("misses.jsonl")).expect("misses");
        let ours = disp
            .lines()
            .filter_map(|l| serde_json::from_str::<DispatchRecord>(l).ok())
            .find(|r| r.structure_key.as_ref().map(|t| t.0.as_str()) == Some(uniq.as_str()))
            .expect("our dispatch record is present in the flushed feed");
        assert_eq!(ours.chosen, impl_id("portable-cpu"));
        assert_eq!(ours.candidates[0].latency_ns, Some(41_230));
        assert!(
            miss.lines()
                .filter_map(|l| serde_json::from_str::<MissRecord>(l).ok())
                .any(|r| r.wanted.0 == uniq),
            "our miss record is present in the flushed feed",
        );
    }

    /// The CPU device's hardware stamp carries no compute capability (the
    /// stampless-CUDA-row case Baracuda's merge drops; CPU rows kept for our
    /// own analysis). Read-only — does not touch the mode/sink globals.
    #[test]
    fn cpu_hw_stamp_has_no_compute_capability() {
        let stamp = hw_stamp_for(DeviceLocation::Cpu);
        assert_eq!(
            stamp.compute_capability, None,
            "a CPU realize has no CUDA compute capability",
        );
    }

    /// `default_telemetry_dir` never panics and is a plain projection of the
    /// Judge report dir (may be `None` in a restricted environment).
    #[test]
    fn default_telemetry_dir_matches_the_profile_cache_dir() {
        let dir = default_telemetry_dir();
        if let Some(dir) = dir {
            let report = crate::judge::default_report_path().expect("report path present");
            assert_eq!(Some(dir.as_path()), report.parent());
        }
    }
}
