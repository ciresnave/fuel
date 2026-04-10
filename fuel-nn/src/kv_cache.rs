//! KV-Cache implementations — re-exported from `fuel_core::kv_cache`.
//!
//! This module is a compatibility shim. The canonical source is
//! `fuel_core::kv_cache`. All types are re-exported here unchanged so
//! that existing code using `fuel_nn::kv_cache::*` continues to compile
//! without modification.
pub use fuel::kv_cache::*;
