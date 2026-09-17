// SPDX-License-Identifier: MIT OR Apache-2.0
//! Adapter selection for the `*_live` tests (GAP-325).
//!
//! `VulkanBackend::new()` prefers the discrete card, so on a box with both a
//! discrete and an integrated GPU every live test ran on the discrete one, and
//! nothing in a green run said so. This is a cross-vendor backend, and a
//! same-vendor pass is not a cross-vendor pass.
//!
//! Set `FUEL_VULKAN_TEST_ADAPTER` to a case-insensitive substring of ONE
//! adapter's name to run the live tests on that adapter.
//! `adapter_inventory_live` prints the names.
//!
//! Unset, nothing changes: `PreferDiscrete`, and a box with no Vulkan device
//! skips. SET, failing to find or open the named adapter PANICS. A requested
//! cross-vendor run must never skip into a pass, and must never fall back to
//! another adapter.

use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

/// The environment variable that names the adapter.
pub const ADAPTER_ENV: &str = "FUEL_VULKAN_TEST_ADAPTER";

/// The backend a live test should run on, or `None` to skip.
///
/// `None` is possible only when [`ADAPTER_ENV`] is unset.
pub fn backend_or_skip() -> Option<VulkanBackend> {
    match std::env::var(ADAPTER_ENV) {
        Ok(needle) => Some(requested(&needle)),
        Err(_) => match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("no Vulkan device; skipping: {e:?}");
                None
            }
        },
    }
}

/// Open the one adapter whose name contains `needle`, or panic.
///
/// Selects by INDEX from `list_devices()`, which enumerates the same way
/// `with_selection` does, so the adapter printed here is the adapter opened.
/// An ambiguous needle is refused rather than resolved to the first match.
fn requested(needle: &str) -> VulkanBackend {
    let listed = VulkanBackend::list_devices()
        .unwrap_or_else(|e| panic!("{ADAPTER_ENV}={needle:?}: list_devices failed: {e:?}"));
    let lower = needle.to_lowercase();
    let hits: Vec<&(usize, String, String)> = listed
        .iter()
        .filter(|(_, name, _)| name.to_lowercase().contains(&lower))
        .collect();
    let (idx, name, kind) = match hits.as_slice() {
        [one] => (one.0, one.1.clone(), one.2.clone()),
        [] => panic!("{ADAPTER_ENV}={needle:?} matches no adapter; inventory: {listed:?}"),
        many => panic!(
            "{ADAPTER_ENV}={needle:?} matches {} adapters {many:?}; use a longer substring",
            many.len()
        ),
    };
    // The vendor, from the probe's own enumeration, so a report can cite an id
    // and not just a marketing name.
    let vendor = fuel_vulkan_backend::probe::enumerate_devices()
        .ok()
        .and_then(|ds| ds.into_iter().find(|d| d.hardware_sku == name))
        .map(|d| format!("vendor_id=0x{:04x}", d.vendor_id))
        .unwrap_or_else(|| "vendor_id=<not matched by the probe>".to_string());
    eprintln!("live-test adapter: [{idx}] {kind} {name} {vendor} ({ADAPTER_ENV}={needle:?})");
    VulkanBackend::with_selection(DeviceSelection::Index(idx)).unwrap_or_else(|e| {
        panic!("{ADAPTER_ENV}={needle:?}: opening adapter [{idx}] {name} failed: {e:?}")
    })
}
