// SPDX-License-Identifier: MIT OR Apache-2.0
//! # fuel-onnx
//!
//! **Layer**: IO — provides bidirectional ONNX interchange for the Fuel stack.
//!
//! **Stability**: `evolving` — operator coverage grows with each release; not all
//! ONNX opsets are supported yet.
//!
//! ## What this crate is for
//!
//! `fuel-onnx` loads ONNX model files and evaluates them against Fuel tensors:
//!
//! - [`read_file`]: deserialize an `.onnx` file into an in-memory `ModelProto`.
//! - [`OnnxEval`]: evaluate an ONNX graph onto the **lazy** graph, producing
//!   [`fuel::lazy::Tensor`] outputs that realize on demand.
//!
//! The eager evaluator (`eval.rs` / `simple_eval`) was **deleted in B6** along
//! with the eager `Tensor` it was built on. `OnnxEval` replaces it.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use std::collections::HashMap;
//! use fuel_onnx::OnnxEval;
//! # fn main() -> fuel::Result<()> {
//! let eval = OnnxEval::from_path("path/to/model.onnx")?;
//! let inputs = HashMap::new(); // populate with fuel::lazy::Tensor values
//! let outputs = eval.run(&inputs)?;
//! # Ok(()) }
//! ```
//!
//! ## What is explicitly NOT here
//!
//! - **No training.** ONNX graphs are evaluated in inference mode only.
//! - **No tokenization.** Textual pre/post-processing is outside scope.
//! - **No model download.** Provide the path to a local `.onnx` file.
//!
//! ## Ecosystem crates
//!
//! - [`fuel-core`](https://docs.rs/fuel-core): tensor primitives used by outputs.
//! - [`fuel-transformers`](https://docs.rs/fuel-transformers): native model
//!   implementations that do not require ONNX export.
//!

use fuel::Result;
use prost::Message;

pub mod onnx {
    // prost-generated from `onnx.proto3` by build.rs. The doc comments are
    // copied verbatim from the .proto, so their list indentation is ONNX's,
    // not ours, and it is regenerated on every build -- there is no source
    // file here to fix. Scoped to this module so it cannot mask our own docs.
    #![allow(clippy::doc_overindented_list_items)]
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub mod lazy_eval;
pub mod lazy_eval_conv;
pub mod lazy_eval_norm;
pub mod lazy_eval_ops;
pub use lazy_eval::{onnx_dtype_to_fuel as dtype, OnnxEval};

/// Reads and deserializes an ONNX model from a file on disk.
///
/// The file is expected to be a protobuf-encoded `ModelProto` (standard `.onnx` format).
///
/// # Example
///
/// ```no_run
/// use fuel_onnx::read_file;
///
/// let model = read_file("path/to/model.onnx")?;
/// let graph = model.graph.as_ref().expect("model has no graph");
/// println!("graph inputs: {}", graph.input.len());
/// # Ok::<(), fuel::Error>(())
/// ```
pub fn read_file<P: AsRef<std::path::Path>>(p: P) -> Result<onnx::ModelProto> {
    let buf = std::fs::read(p)?;
    onnx::ModelProto::decode(buf.as_slice()).map_err(fuel::Error::wrap)
}
