// SPDX-License-Identifier: MIT OR Apache-2.0
//! The on-disk path for the Judge's persisted [`fuel_ir::dispatch::ProfileReport`].
//!
//! Moved from `fuel-core::judge::mod::default_report_path` (fuel-core
//! dissolution, plan item 8): pure path-joining logic with zero coupling to
//! `fuel-core`'s `Tensor`/`Device` or to the Judge struct itself, so it moves
//! independently of the rest of `judge/mod.rs` (which stays in fuel-core).
//! Kept UNGATED (not under the `telemetry` feature) because
//! `fuel-core::judge::cache` and `fuel-core::scheduling` call it
//! unconditionally, regardless of whether `telemetry` is enabled.

/// Filename of the persisted profile report, alongside the probe report in
/// the same hardware-keyed cache directory.
pub const PROFILE_REPORT_FILENAME: &str = "judge.json";

/// The default on-disk path for the Judge's persisted profile report —
/// the same hardware-keyed cache directory [`fuel_hardware::probe`] uses,
/// with the profile report's own filename joined on.
pub fn default_report_path() -> Option<std::path::PathBuf> {
    fuel_hardware::probe::default_report_path().and_then(|p| {
        p.parent()
            .map(|parent| parent.join(PROFILE_REPORT_FILENAME))
    })
}
