// SPDX-License-Identifier: MIT OR Apache-2.0
//! Moved to the standalone [`fuel_kv_pool`] crate (fuel-core dissolution,
//! plan item 9): a pure host-side data structure with zero coupling to
//! `Tensor`/`Device`, extracted to a leaf crate so `fuel-model-llama` (a
//! real consumer, itself a dependency OF `fuel-inference`) doesn't need a
//! cycle through `fuel-inference` to reach it. Re-exported here so
//! `fuel_core::kv_block_pool` / `fuel::kv_block_pool` call sites
//! (`kv_block_pool_device.rs`, `fuel-model-llama`, `fuel-inference`, and
//! the lightbulb sibling project) are unchanged.
//!
//! TRANSITIONAL: this shim dies with `fuel-core` itself. Every listed
//! consumer should repoint to `fuel_kv_pool` directly before `fuel-core`
//! is deleted — tracked, not done here, per PM direction.
pub use fuel_kv_pool::*;
