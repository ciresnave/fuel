//! Phase 6c-A demo: probe → measure transfer-cost matrix → print.
//!
//! USAGE
//!
//!     cargo run --release --bin bandwidth
//!     cargo run --release --bin bandwidth --features cuda
//!     cargo run --release --bin bandwidth --features "cuda vulkan"

use fuel::probe::ProbeReport;
use fuel::transfer_cost::BandwidthMatrix;

fn main() {
    let probe = ProbeReport::probe_all();
    eprintln!(
        "Probed {} device(s) across {} equivalence class(es).",
        probe.devices.len(),
        probe.equivalence_classes().len(),
    );
    eprintln!("Measuring bandwidth matrix (this takes a few seconds)...");

    let t0 = std::time::Instant::now();
    let m = BandwidthMatrix::measure(&probe);
    eprintln!(
        "Done in {:.2?}. Buffer size = {} MiB; {} entries.",
        t0.elapsed(),
        m.measurement_bytes / (1 << 20),
        m.entries.len(),
    );
    println!();

    println!("Transfer-cost matrix (ns/byte; lower = faster):");
    println!("{:<14}  {:<14}  {:>14}  {:>16}", "src", "dst", "ns/byte", "GB/s effective");
    println!("{}", "-".repeat(64));
    for e in &m.entries {
        // GB/s = 1e9 / (ns_per_byte * 1e9 / 1e9) = 1.0 / ns_per_byte
        let gbs = if e.ns_per_byte > 0.0 { 1.0 / e.ns_per_byte } else { f64::INFINITY };
        println!(
            "{:<14}  {:<14}  {:>14.4}  {:>16.2}",
            e.src.to_string(), e.dst.to_string(), e.ns_per_byte, gbs,
        );
    }
}
