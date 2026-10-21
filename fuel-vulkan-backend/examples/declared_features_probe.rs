// SPDX-License-Identifier: MIT OR Apache-2.0
//! Prints, for every Vulkan device, which optional features the committed SPIR-V
//! declares and whether the device reports them (`probe_optional_features`), plus
//! the backend's resulting `declared_features_absent` list. A device that lacks one
//! still gets a backend (the feature is simply not enabled).
//!
//! ```text
//! pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release \
//!     -p fuel-vulkan-backend --example declared_features_probe
//! ```

use fuel_vulkan_backend::{DeviceSelection, VulkanBackend, probe_optional_features};

fn main() {
    let mut index = 0;
    while let Ok(be) = VulkanBackend::with_selection(DeviceSelection::Index(index)) {
        println!(
            "[{index}] {}\n    reported: {:?}\n    absent (not enabled): {:?}",
            be.device_name,
            probe_optional_features(&be.physical),
            be.declared_features_absent
        );
        index += 1;
    }
    println!("{index} device(s)");
}
