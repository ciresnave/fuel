//! End-to-end Phase 6b demo: probe → load-or-judge → dispatch table →
//! recommend per-node placement on a small graph.
//!
//! USAGE
//!
//!     cargo run --release --bin place
//!     cargo run --release --bin place --features cuda
//!     cargo run --release --bin place --features "cuda vulkan"
//!
//! The graph mixes a few matmul sizes and a couple of unprofiled
//! ops so you can see the dispatch-table-driven recommendations and
//! the fallback-to-default behaviour side by side.

use fuel::dispatch::Criterion;
use fuel::scheduling::{prepare_dispatch_table, recommend_placement, ScheduleOptions};
use fuel_core_types::{DeviceLocation, Shape};
use fuel_graph::Tensor;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Phase 6b end-to-end demo");
    eprintln!("========================");

    // Step 1: probe → judge → dispatch table.
    // Honours the persistent cache; if you've run `judge` once, this is fast.
    let t0 = std::time::Instant::now();
    let (table, profile) = prepare_dispatch_table(ScheduleOptions::default())?;
    eprintln!(
        "Loaded dispatch table ({} entries from {} profile measurements) in {:.2?}",
        table.len(), profile.entries.len(), t0.elapsed(),
    );
    eprintln!();

    // Step 2: build a heterogeneous graph — three matmul sizes plus
    // a couple of unprofiled ops (Sub, Silu).
    let root = Tensor::from_f32(vec![0.0_f32; 64 * 64], Shape::from_dims(&[64, 64]));
    let mk = |elems: usize| Arc::<[f32]>::from(vec![0.0_f32; elems]);

    // Tiny matmul: 64×64 @ 64×64 (size_class 12)
    let tiny_a = root.clone();
    let tiny_b = root.const_f32_like(mk(64 * 64), Shape::from_dims(&[64, 64]));
    let tiny_mm = tiny_a.matmul(&tiny_b);

    // Medium matmul: 256×256 @ 256×256 (size_class 16)
    let mid_a = root.const_f32_like(mk(256 * 256), Shape::from_dims(&[256, 256]));
    let mid_b = root.const_f32_like(mk(256 * 256), Shape::from_dims(&[256, 256]));
    let mid_mm = mid_a.matmul(&mid_b);

    // Large matmul: 1024×1024 @ 1024×1024 (size_class 20)
    let big_a = root.const_f32_like(mk(1024 * 1024), Shape::from_dims(&[1024, 1024]));
    let big_b = root.const_f32_like(mk(1024 * 1024), Shape::from_dims(&[1024, 1024]));
    let big_mm = big_a.matmul(&big_b);

    // Unprofiled ops — Sub and Silu — should use fallback.
    let unprofiled_sub = tiny_mm.sub(&tiny_b);
    let unprofiled_silu = tiny_mm.silu();

    // Step 3: recommend placement under each criterion.
    let graph = root.graph().borrow();
    let fallback = DeviceLocation::Cpu;

    println!("Per-node placement (graph has {} nodes):", graph.len());
    println!();
    for &criterion in &[Criterion::Fastest, Criterion::MostAccurate, Criterion::Balanced] {
        let plan = recommend_placement(&graph, &table, criterion, fallback);
        println!("=== Criterion: {criterion} ===");
        for (label, t) in [
            ("tiny  matmul (64×64²)",      &tiny_mm),
            ("mid   matmul (256×256²)",    &mid_mm),
            ("large matmul (1024×1024²)",  &big_mm),
            ("unprofiled sub",             &unprofiled_sub),
            ("unprofiled silu",            &unprofiled_silu),
        ] {
            println!("  {label:<28}  →  {:?}", plan[&t.id()]);
        }
        println!();
    }

    Ok(())
}
