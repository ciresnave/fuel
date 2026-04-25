//! Enumerate every device Fuel's backends can reach and print a
//! human-readable report plus the equivalence-class summary.
//!
//! USAGE
//!
//!     cargo run --release --bin probe
//!     cargo run --release --bin probe --features cuda
//!     cargo run --release --bin probe --features "cuda vulkan"
//!
//! Optional first arg saves the JSON report to that path.

use fuel::probe::ProbeReport;

fn main() {
    let report = ProbeReport::probe_all();
    println!("Probe report (version {}):", report.version);
    println!("  {} device(s) total", report.devices.len());
    println!();
    for d in &report.devices {
        let cc = d.compute_capability
            .map(|(a, b)| format!("sm_{a}{b}"))
            .unwrap_or_else(|| "n/a".to_string());
        let mem_gib = d.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        println!(
            "  [{}:{}] {}  (vendor 0x{:04x}  device 0x{:04x}  cc={}  driver={}  vram={:.1} GiB)",
            d.backend,
            d.device_index,
            d.hardware_sku,
            d.vendor_id,
            d.device_id,
            cc,
            d.driver_version,
            mem_gib,
        );
    }
    println!();
    let classes = report.equivalence_classes();
    println!("Equivalence classes ({} distinct):", classes.len());
    for (key, devs) in &classes {
        println!(
            "  backend={} vendor=0x{:04x} device=0x{:04x} driver={} → {} device(s)",
            key.backend, key.vendor_id, key.device_id, key.driver_version, devs.len(),
        );
    }
    if let Some(path) = std::env::args().nth(1) {
        report.save(std::path::Path::new(&path)).expect("save");
        println!();
        println!("Wrote JSON report to {path}");
    } else if let Some(default) = fuel::probe::default_report_path() {
        println!();
        println!("(Default persist path would be: {})", default.display());
    }
}
