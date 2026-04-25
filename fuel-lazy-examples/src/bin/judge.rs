//! Run the Judge: profile every op × dtype × size cell on every
//! (backend, device) the probe discovered. Prints a table and, if
//! given a path, writes the JSON profile report.
//!
//! USAGE
//!
//!     cargo run --release --bin judge
//!     cargo run --release --bin judge --features cuda
//!     cargo run --release --bin judge --features "cuda vulkan" -- /tmp/judge.json
//!
//! The table's "latency" column is the median of `iterations` runs
//! (default 7) after `warmup` throwaway runs (default 3). "rel err"
//! is max element-wise relative error vs the reference backend's
//! output on the same input.

use fuel::dispatch::{Criterion, DispatchTable};
use fuel::judge::{Judge, ProfileEntry};
use fuel::probe::ProbeReport;

fn main() {
    eprintln!("Probing devices...");
    let probe = ProbeReport::probe_all();
    eprintln!("  {} device(s) total, {} equivalence class(es)",
        probe.devices.len(),
        probe.equivalence_classes().len(),
    );
    eprintln!();

    eprintln!("Running Judge (this may take a minute)...");
    let judge = Judge::default();
    let t0 = std::time::Instant::now();
    let report = judge.run(&probe);
    eprintln!("  {} profile entries in {:.2?}", report.entries.len(), t0.elapsed());
    eprintln!();

    print_table(&report.entries);
    println!();
    print_dispatch_summary(&report);

    if let Some(path) = std::env::args().nth(1) {
        report.save(std::path::Path::new(&path)).expect("save");
        eprintln!();
        eprintln!("Wrote profile report to {path}");
    } else if let Some(default) = fuel::judge::default_report_path() {
        eprintln!();
        eprintln!("(Default persist path would be: {})", default.display());
    }
}

fn print_table(entries: &[ProfileEntry]) {
    // Sort entries by (op, size_class, backend) so rows group by op
    // and size class — human-readable rather than the dispatch-
    // friendly order the Judge emits.
    let mut sorted: Vec<&ProfileEntry> = entries.iter().collect();
    sorted.sort_by(|a, b| {
        a.op.as_str().cmp(b.op.as_str())
            .then(a.size_class.0.cmp(&b.size_class.0))
            .then(a.backend.as_str().cmp(b.backend.as_str()))
            .then(a.device_index.cmp(&b.device_index))
    });

    println!("{:<10}  {:<5}  {:<6}  {:<12}  {:>14}  {:>10}", "op", "dtype", "2^n",
        "backend:dev", "latency", "rel err");
    println!("{}", "-".repeat(72));
    for e in &sorted {
        let latency_str = human_ns(e.latency_ns);
        println!(
            "{:<10}  {:<5}  {:<6}  {:<12}  {:>14}  {:>10.2e}",
            e.op.to_string(),
            format!("{:?}", e.dtype),
            e.size_class.0,
            format!("{}:{}", e.backend, e.device_index),
            latency_str,
            e.max_rel_error,
        );
    }
}

fn print_dispatch_summary(report: &fuel::judge::ProfileReport) {
    let tbl = DispatchTable::build(report);
    println!("Dispatch winners (reference excluded):");
    println!("{:<10}  {:<5}  {:<4}  {:<14}  {:<14}  {:<14}", "op", "dtype",
        "2^n", "fastest", "accurate", "balanced");
    println!("{}", "-".repeat(72));
    for (op, dtype, size) in tbl.keys() {
        let pick = |c| tbl.pick(op, dtype, size, c)
            .map(|p| format!("{}:{}", p.backend, p.device_index))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<10}  {:<5}  {:<4}  {:<14}  {:<14}  {:<14}",
            op.to_string(),
            format!("{:?}", dtype),
            size.0,
            pick(Criterion::Fastest),
            pick(Criterion::MostAccurate),
            pick(Criterion::Balanced),
        );
    }
}

fn human_ns(ns: u64) -> String {
    if ns < 10_000                      { format!("{} ns",  ns) }
    else if ns < 10_000_000             { format!("{:.2} μs", ns as f64 / 1_000.0) }
    else if ns < 10_000_000_000         { format!("{:.2} ms", ns as f64 / 1_000_000.0) }
    else                                { format!("{:.2} s",  ns as f64 / 1e9) }
}
