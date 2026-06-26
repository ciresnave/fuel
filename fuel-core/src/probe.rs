//! Top-level multi-backend probe collector for Phase 6b.
//!
//! Calls every enabled backend's `enumerate_devices()` and assembles
//! a single [`ProbeReport`] that can be serialised to disk. The report
//! is what the (future) Judge consumes to decide which
//! `(backend, device)` pairs to profile, and it's what the dispatch
//! table uses as its invalidation key — if the current probe doesn't
//! match the previously persisted one, the Judge re-runs and the
//! tables are rebuilt.
//!
//! # Feature gating
//!
//! Backends that aren't compiled in don't contribute to the report.
//! A build without the `cuda` feature produces a report with zero
//! CUDA entries; adding the feature later produces a report with
//! CUDA entries and (via [`ProbeReport::diff`]) signals "devices
//! added" so the Judge re-runs. The reverse is also true: dropping a
//! feature signals "devices removed."
//!
//! # Stability
//!
//! The JSON schema is `{ "version": u32, "devices": [DeviceDescriptor,
//! ...] }`. Bumping `PROBE_REPORT_VERSION` invalidates old persisted
//! reports; this is the escape hatch for when the descriptor schema
//! itself changes in a way that `#[serde(default)]` can't cover.
//! Today: version 1.

use fuel_ir::probe::{BackendId, DeviceDescriptor, EquivalenceKey};
use fuel_ir::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Schema version for persisted probe reports. Bump when the
/// descriptor layout changes in a way that breaks backward
/// compatibility.
pub const PROBE_REPORT_VERSION: u32 = 1;

/// A persistable snapshot of every `(backend, device)` pair Fuel
/// currently has access to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeReport {
    pub version: u32,
    /// All descriptors in stable order: sorted by `(backend.as_str(),
    /// device_index)`. Sorting is enforced at construction; callers
    /// can therefore compare two reports byte-for-byte (or diff with
    /// `Vec::eq`) as a cheap "did anything change?" check.
    pub devices: Vec<DeviceDescriptor>,
}

impl ProbeReport {
    /// Run every enabled backend's `enumerate_devices()` and assemble
    /// a fresh report. Backends whose probes error out are skipped
    /// (with a `tracing::warn!` — the Judge shouldn't be held up by
    /// one backend's driver being half-loaded). Devices are sorted
    /// by `(backend, device_index)` so identical reports compare
    /// equal byte-for-byte.
    pub fn probe_all() -> Self {
        let mut devices = Vec::new();

        // Walk the BackendFactory registry — each compiled-in backend
        // contributes one entry. New backends register themselves in
        // `crate::factories` and show up here automatically.
        for factory in crate::factories::registry() {
            Self::collect(
                &mut devices,
                factory.id().as_str(),
                || factory.enumerate_devices(),
            );
        }

        devices.sort_by(|a, b| {
            a.backend.as_str().cmp(b.backend.as_str())
                .then(a.device_index.cmp(&b.device_index))
        });

        Self { version: PROBE_REPORT_VERSION, devices }
    }

    fn collect(
        devices: &mut Vec<DeviceDescriptor>,
        label: &str,
        enumerator: impl FnOnce() -> Result<Vec<DeviceDescriptor>>,
    ) {
        match enumerator() {
            Ok(mut ds) => devices.append(&mut ds),
            Err(e) => {
                eprintln!(
                    "fuel probe: {label} backend enumerate_devices failed, \
                     skipping: {e}"
                );
            }
        }
    }

    /// Group the report's descriptors by [`EquivalenceKey`]. Identical
    /// devices (e.g. four RTX 4090s in the same rig) end up in the
    /// same bucket; the Judge profiles one representative per bucket
    /// and the dispatch table shares the result across the others.
    pub fn equivalence_classes(&self) -> std::collections::HashMap<EquivalenceKey, Vec<&DeviceDescriptor>> {
        let mut map: std::collections::HashMap<EquivalenceKey, Vec<&DeviceDescriptor>> =
            Default::default();
        for d in &self.devices {
            map.entry(d.equivalence_key()).or_default().push(d);
        }
        map
    }

    /// Serialise to a JSON file at `path`. Overwrites atomically if
    /// possible (writes to a sibling `.tmp` file then renames).
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| fuel_ir::Error::Msg(
                format!("probe: JSON encode failed: {e}")
            ))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &json)
            .map_err(|e| fuel_ir::Error::Msg(
                format!("probe: write {tmp:?} failed: {e}")
            ))?;
        std::fs::rename(&tmp, path)
            .map_err(|e| fuel_ir::Error::Msg(
                format!("probe: rename {tmp:?} → {path:?} failed: {e}")
            ))?;
        Ok(())
    }

    /// Load a previously-persisted report. Returns `Ok(None)` if the
    /// file does not exist (first run — expected, not an error) and
    /// `Err` only on real I/O or JSON parse failures.
    pub fn load(path: &Path) -> Result<Option<Self>> {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(fuel_ir::Error::Msg(
                format!("probe: read {path:?} failed: {e}")
            )),
        };
        let report: Self = serde_json::from_slice(&bytes)
            .map_err(|e| fuel_ir::Error::Msg(
                format!("probe: parse {path:?} failed: {e}")
            ))?;
        if report.version != PROBE_REPORT_VERSION {
            // Version mismatch is not an error — it's a cache miss.
            // Caller treats this the same as "no previous report."
            return Ok(None);
        }
        Ok(Some(report))
    }

    /// Compare this report against a prior one to decide whether
    /// downstream caches (the Judge's profile tables) need to be
    /// invalidated.
    ///
    /// Returns [`HardwareChange::Unchanged`] iff the two reports are
    /// byte-equal. Otherwise describes which descriptors are new,
    /// removed, or changed so callers can take a graduated action
    /// (re-Judge only the affected equivalence classes, not the whole
    /// matrix).
    pub fn diff(&self, prior: &Self) -> HardwareChange {
        if self == prior {
            return HardwareChange::Unchanged;
        }
        let now_keys: std::collections::HashSet<_> =
            self.devices.iter().map(|d| d.equivalence_key()).collect();
        let prior_keys: std::collections::HashSet<_> =
            prior.devices.iter().map(|d| d.equivalence_key()).collect();
        let added: Vec<EquivalenceKey> =
            now_keys.difference(&prior_keys).cloned().collect();
        let removed: Vec<EquivalenceKey> =
            prior_keys.difference(&now_keys).cloned().collect();
        HardwareChange::Changed { added, removed }
    }
}

/// Outcome of a [`ProbeReport::diff`] comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HardwareChange {
    /// The two reports are byte-identical. No re-probing or re-Judging
    /// required; dispatch tables remain valid.
    Unchanged,
    /// Hardware composition shifted. `added` and `removed` list the
    /// equivalence classes that appeared / disappeared. A class that
    /// merely grew in count (e.g. 2 RTX 4090s → 4) does not appear
    /// in either list; the Judge can optionally bump the worker count
    /// but doesn't need to re-measure.
    Changed {
        added:   Vec<EquivalenceKey>,
        removed: Vec<EquivalenceKey>,
    },
}

impl HardwareChange {
    pub fn needs_rejudge(&self) -> bool {
        !matches!(self, HardwareChange::Unchanged)
    }
}

/// Conventional filename for the persisted probe report. Store this
/// under the user's Fuel cache directory.
pub const PROBE_REPORT_FILENAME: &str = "probe.json";

/// Return the default location for the probe report — OS-conventional
/// cache directory with the `fuel/` subdirectory. Returns `None` on
/// systems where `dirs::cache_dir()` would yield `None` (very rare).
/// Callers that want an explicit path should pass one directly to
/// [`ProbeReport::save`] / [`ProbeReport::load`] instead of using
/// this helper.
pub fn default_report_path() -> Option<std::path::PathBuf> {
    // We intentionally avoid a `dirs` crate dep for this one call —
    // the logic is trivial and crate bloat for a single fallback
    // isn't worth it. On Windows, %LOCALAPPDATA%; on *nix,
    // $XDG_CACHE_HOME or ~/.cache.
    let base = std::env::var_os("LOCALAPPDATA")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("XDG_CACHE_HOME").map(std::path::PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|h| {
            let mut p = std::path::PathBuf::from(h);
            p.push(".cache");
            p
        }))?;
    let mut p = base;
    p.push("fuel");
    p.push(PROBE_REPORT_FILENAME);
    Some(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::DeviceLocation;

    fn sample_cuda_descriptor(idx: u32) -> DeviceDescriptor {
        DeviceDescriptor {
            backend:            BackendId::Cuda,
            device_index:       idx,
            hardware_sku:       "NVIDIA GeForce RTX 4090".to_string(),
            vendor_id:          0x10DE,
            device_id:          0x2684,
            compute_capability: Some((8, 9)),
            driver_version:     "CUDA 12.6".to_string(),
            total_memory_bytes: 25_769_803_776,
            location:           DeviceLocation::Cuda { gpu_id: idx as usize },
        }
    }

    #[test]
    fn probe_all_runs_total() {
        // Regardless of which backends are compiled in, this should
        // never panic and should always include at least the CPU
        // descriptor. (The Reference backend was retired 2026-06-07;
        // CPU is the only unconditionally-present backend now.)
        let report = ProbeReport::probe_all();
        assert_eq!(report.version, PROBE_REPORT_VERSION);
        assert!(
            report.devices.iter().any(|d| d.backend == BackendId::Cpu),
            "cpu should always be present, got {:?}", report.devices);
        // Verify sort invariant.
        for pair in report.devices.windows(2) {
            let a = &pair[0];
            let b = &pair[1];
            let order = a.backend.as_str().cmp(b.backend.as_str())
                .then(a.device_index.cmp(&b.device_index));
            assert!(
                order != std::cmp::Ordering::Greater,
                "probe report not sorted: {} {} before {} {}",
                a.backend, a.device_index, b.backend, b.device_index,
            );
        }
    }

    #[test]
    fn save_load_roundtrip() {
        let tmp = std::env::temp_dir().join(format!(
            "fuel-probe-test-{}.json", std::process::id()
        ));
        let original = ProbeReport {
            version: PROBE_REPORT_VERSION,
            devices: vec![sample_cuda_descriptor(0), sample_cuda_descriptor(1)],
        };
        original.save(&tmp).expect("save");
        let loaded = ProbeReport::load(&tmp)
            .expect("load")
            .expect("file exists");
        assert_eq!(loaded, original);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn load_missing_file_returns_none() {
        let tmp = std::env::temp_dir().join(format!(
            "fuel-probe-test-nonexistent-{}.json", std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        let loaded = ProbeReport::load(&tmp).expect("missing file is not an error");
        assert!(loaded.is_none());
    }

    #[test]
    fn load_wrong_version_returns_none() {
        let tmp = std::env::temp_dir().join(format!(
            "fuel-probe-test-oldver-{}.json", std::process::id()
        ));
        let ancient = serde_json::json!({
            "version": 0,
            "devices": [],
        });
        std::fs::write(&tmp, serde_json::to_vec(&ancient).unwrap()).unwrap();
        let loaded = ProbeReport::load(&tmp).expect("old version parses, not errors");
        assert!(loaded.is_none(), "old-version reports should be treated as cache miss");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn diff_identical_reports_unchanged() {
        let report = ProbeReport::probe_all();
        assert_eq!(report.diff(&report), HardwareChange::Unchanged);
        assert!(!report.diff(&report).needs_rejudge());
    }

    #[test]
    fn diff_detects_added_device_class() {
        let empty = ProbeReport { version: PROBE_REPORT_VERSION, devices: vec![] };
        let with_cuda = ProbeReport {
            version: PROBE_REPORT_VERSION,
            devices: vec![sample_cuda_descriptor(0)],
        };
        let change = with_cuda.diff(&empty);
        match change {
            HardwareChange::Changed { added, removed } => {
                assert_eq!(added.len(), 1, "one class added");
                assert!(removed.is_empty(), "nothing removed");
                assert!(added[0].backend == BackendId::Cuda);
            }
            HardwareChange::Unchanged => panic!("expected a change"),
        }
    }

    #[test]
    fn diff_detects_removed_device_class() {
        let with_cuda = ProbeReport {
            version: PROBE_REPORT_VERSION,
            devices: vec![sample_cuda_descriptor(0)],
        };
        let empty = ProbeReport { version: PROBE_REPORT_VERSION, devices: vec![] };
        let change = empty.diff(&with_cuda);
        match change {
            HardwareChange::Changed { added, removed } => {
                assert!(added.is_empty());
                assert_eq!(removed.len(), 1);
            }
            HardwareChange::Unchanged => panic!("expected a change"),
        }
    }

    #[test]
    fn diff_ignores_count_within_equivalence_class() {
        // Four identical RTX 4090s then two — the equivalence class
        // is still present, so the Judge doesn't need to re-run. No
        // added / removed equivalence keys.
        let four = ProbeReport {
            version: PROBE_REPORT_VERSION,
            devices: (0..4).map(sample_cuda_descriptor).collect(),
        };
        let two = ProbeReport {
            version: PROBE_REPORT_VERSION,
            devices: (0..2).map(sample_cuda_descriptor).collect(),
        };
        match two.diff(&four) {
            HardwareChange::Changed { added, removed } => {
                assert!(added.is_empty(), "class count change → no added keys");
                assert!(removed.is_empty(), "class count change → no removed keys");
            }
            HardwareChange::Unchanged => panic!("reports not byte-equal, expected Changed"),
        }
    }

    #[test]
    fn equivalence_classes_groups_identical_devices() {
        let report = ProbeReport {
            version: PROBE_REPORT_VERSION,
            devices: (0..4).map(sample_cuda_descriptor).collect(),
        };
        let classes = report.equivalence_classes();
        assert_eq!(classes.len(), 1, "four identical 4090s → one class");
        assert_eq!(classes.values().next().unwrap().len(), 4);
    }
}
