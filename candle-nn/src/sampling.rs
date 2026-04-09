//! Gumbel-softmax sampling — re-exported from `candle_core::sampling`.
//!
//! This module is a compatibility shim. The canonical source is
//! `candle_core::sampling`. All items are re-exported here unchanged so that
//! existing code using `candle_nn::sampling::*` continues to compile without
//! modification.
pub use candle::sampling::*;
