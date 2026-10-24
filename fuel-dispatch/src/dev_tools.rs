// SPDX-License-Identifier: MIT OR Apache-2.0
//! Maintenance tools behind the `dev-tools` feature. **Not part of the
//! supported API; semver-exempt.**
//!
//! These were `#[ignore]`d tests: each one REWRITES a committed file (the
//! verified ledger, or the CPU contracts), so they are tools, not assertions.
//! A test that is ignored on purpose reads as flaky or slow and cannot say what
//! it really is. They live here so the `fuel-dev-tools` binary (a separate
//! crate, unable to reach `pub(crate)` internals) can drive them, and so
//! `cargo check --features dev-tools --bins` keeps them compiling.

use crate::fkc::{contract_audit_flip, verify};

/// Re-seed the CPU half of `fuel-dispatch/kernel-contracts/fkc-verified-ledger.json`.
pub fn seed_cpu_ledger() {
    verify::seed_cpu_ledger::run_seed_cpu_ledger();
}

/// Re-seed the CUDA half of the verified ledger. Needs a live CUDA device.
#[cfg(feature = "cuda")]
pub fn seed_cuda_ledger() {
    verify::seed_cuda_ledger::run_seed_cuda_ledger();
}

/// Earn the CUDA f32 matmul's `max_relative`/`max_absolute` ledger records (ROADMAP row 39).
/// Needs a live CUDA device.
#[cfg(feature = "cuda")]
pub fn seed_cuda_accuracy() {
    verify::seed_cuda_accuracy::run_seed_cuda_accuracy();
}

/// Measure the CUDA f32 accuracy claim and print; writes nothing (ROADMAP row 39). Needs a live
/// CUDA device.
#[cfg(feature = "cuda")]
pub fn check_cuda_accuracy() {
    verify::seed_cuda_accuracy::run_check_cuda_accuracy();
}

/// Re-seed the Vulkan half of the verified ledger. Needs a live Vulkan device.
#[cfg(feature = "vulkan")]
pub fn seed_vulkan_ledger() {
    verify::seed_vulkan_ledger::run_seed_vulkan_ledger();
}

/// Earn the Vulkan f32 matmul's `max_relative`/`max_absolute` ledger records (board 240).
/// Needs a live Vulkan device.
#[cfg(feature = "vulkan")]
pub fn seed_vulkan_matmul_accuracy() {
    verify::seed_vulkan_matmul_accuracy::run_seed_vulkan_matmul_accuracy();
}

/// Measure the Vulkan f32 accuracy claims and print; writes nothing (board 240). Needs a live
/// Vulkan device; `FUEL_VULKAN_TEST_ADAPTER` selects the adapter.
#[cfg(feature = "vulkan")]
pub fn check_vulkan_accuracy() {
    verify::seed_vulkan_matmul_accuracy::run_check_vulkan_accuracy();
}

/// GAP-228: flip the fully-backed `audited: false` CPU contract sections.
pub fn gap_228_flip() {
    contract_audit_flip::run_gap_228_flip();
}

/// GAP-228: report which sections are wholesale flippable (read-only).
pub fn gap_228_sections_report() {
    contract_audit_flip::run_gap_228_sections_report();
}
