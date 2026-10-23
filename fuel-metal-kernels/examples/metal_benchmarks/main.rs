// SPDX-License-Identifier: MIT OR Apache-2.0
//! Simple Metal benchmarks, easy to run under perf etc. The benchmark body is
//! Apple-only (`fuel-metal-kernels` is an empty crate elsewhere), so on any other
//! host this example is a stub: Cargo still discovers examples on every target,
//! and an unguarded import of the crate's API would break `--all-targets`.

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod apple;

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn main() -> anyhow::Result<()> {
    apple::main()
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn main() {
    eprintln!("metal_benchmarks requires macOS or iOS");
}
