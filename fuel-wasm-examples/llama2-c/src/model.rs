//! Llama2-C model wrapper used by the WASM example.
//!
//! Re-exports the lazy [`Llama2cModel`] from `fuel_core::lazy_llama2c`.
//! The eager `fuel_nn`-based custom transformer implementation that used to
//! live here was retired alongside the eager `fuel_nn` + `fuel_transformers::models`
//! crates; the lazy port in `fuel_core::lazy_llama2c` is now the single
//! source of truth for the llama2.c architecture and its custom binary
//! checkpoint loader (`load_llama2c_bin`).

pub use fuel::lazy_llama2c::{
    Llama2cConfig as Config, Llama2cModel as Llama, load_llama2c_bin,
};
