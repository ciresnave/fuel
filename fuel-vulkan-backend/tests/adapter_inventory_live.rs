// SPDX-License-Identifier: MIT OR Apache-2.0
//! Prints the Vulkan adapter inventory this box actually exposes.
//!
//! WHY THIS EXISTS. `VulkanBackend::new()` uses `DeviceSelection::PreferDiscrete`
//! and hands back the discrete card. CLAUDE.md records the consequence as a
//! standing trap: a test that means to cover two vendors "silently becomes a
//! same-vendor test that still passes". Nothing in a green live-GPU run says
//! WHICH adapter produced it, so every such green is reported against an
//! assumed device.
//!
//! `PreferDiscrete should give NVIDIA` is an INFERENCE. The name in this
//! output is a MEASUREMENT. This test exists so a report can cite the second.
//!
//! It asserts almost nothing on purpose — it is an instrument, not a gate. The
//! one thing it does assert is that enumeration returned at least one adapter,
//! so that an empty inventory fails loudly instead of printing nothing and
//! passing (a live-GPU test that cannot run must say so, never report `ok`).

#[test]
#[ignore = "live GPU: enumerates real Vulkan adapters; run via scripts/gpu-run.ps1"]
fn print_the_adapter_inventory() {
    let listed = fuel_vulkan_backend::VulkanBackend::list_devices()
        .expect("list_devices failed on a box that is supposed to have Vulkan");

    println!("=== VULKAN ADAPTER INVENTORY (measured) ===");
    for (idx, name, kind) in &listed {
        println!("  [{idx}] {kind:<10} {name}");
    }

    let descriptors =
        fuel_vulkan_backend::probe::enumerate_devices().expect("probe enumeration failed");
    println!("=== PROBE DESCRIPTORS (vendor/device identity) ===");
    for d in &descriptors {
        println!(
            "  {} vendor_id=0x{:04x} device_id=0x{:04x}",
            d.hardware_sku, d.vendor_id, d.device_id
        );
    }

    // Which adapter `PreferDiscrete` yields is DERIVABLE from this inventory —
    // the selector takes the FIRST `discrete` and stops — but only when the
    // count of discrete adapters is unambiguous. Print the count so a reader
    // can see whether the derivation is sound rather than assume it.
    let discrete: Vec<&String> = listed
        .iter()
        .filter(|(_, _, k)| k == "discrete")
        .map(|(_, n, _)| n)
        .collect();
    println!(
        "=== discrete adapters: {} {:?} ===",
        discrete.len(),
        discrete
    );
    println!(
        "=== PreferDiscrete therefore selects: {} ===",
        match discrete.len() {
            1 => discrete[0].clone(),
            0 => "<none discrete; falls back to first non-CPU/OTHER>".to_string(),
            _ => "<AMBIGUOUS: more than one discrete adapter>".to_string(),
        }
    );

    assert!(
        !listed.is_empty(),
        "no Vulkan adapters enumerated — this test cannot report an inventory, \
         and must fail rather than pass having printed nothing"
    );
}
