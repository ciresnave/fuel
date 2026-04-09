//! KV-Cache implementations — re-exported from `candle_core::kv_cache`.
//!
//! This module is a compatibility shim. The canonical source is
//! `candle_core::kv_cache`. All types are re-exported here unchanged so
//! that existing code using `candle_nn::kv_cache::*` continues to compile
//! without modification.
pub use candle::kv_cache::*;
