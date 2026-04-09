//! # candle-onnx
//!
//! **Layer**: IO — provides bidirectional ONNX interchange for the Candle stack.
//!
//! **Stability**: `evolving` — operator coverage grows with each release; not all
//! ONNX opsets are supported yet.
//!
//! ## What this crate is for
//!
//! `candle-onnx` loads ONNX model files and evaluates them against Candle tensors:
//!
//! - [`read_file`]: deserialize an `.onnx` file into an in-memory `ModelProto`.
//! - [`simple_eval`]: evaluate an ONNX graph given a map of input tensors; returns
//!   a map of output tensors.
//! - [`dtype`]: convert an ONNX data type integer to a `candle_core::DType`.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use std::collections::HashMap;
//! use candle_onnx::{read_file, simple_eval};
//! # fn main() -> candle_core::Result<()> {
//! let model = read_file("path/to/model.onnx")?;
//! let graph = model.graph.as_ref().expect("model has no graph");
//! let inputs = HashMap::new(); // populate with candle_core::Tensor values
//! let outputs = simple_eval(&model, inputs)?;
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
//! - [`candle-core`](https://docs.rs/candle-core): tensor primitives used by outputs.
//! - [`candle-transformers`](https://docs.rs/candle-transformers): native model
//!   implementations that do not require ONNX export.
//!

use candle::Result;
use prost::Message;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub mod eval;
pub use eval::{dtype, simple_eval};

/// Reads and deserializes an ONNX model from a file on disk.
///
/// The file is expected to be a protobuf-encoded `ModelProto` (standard `.onnx` format).
///
/// # Example
///
/// ```no_run
/// use candle_onnx::read_file;
///
/// let model = read_file("path/to/model.onnx")?;
/// let graph = model.graph.as_ref().expect("model has no graph");
/// println!("graph inputs: {}", graph.input.len());
/// # Ok::<(), candle::Error>(())
/// ```
pub fn read_file<P: AsRef<std::path::Path>>(p: P) -> Result<onnx::ModelProto> {
    let buf = std::fs::read(p)?;
    onnx::ModelProto::decode(buf.as_slice()).map_err(candle::Error::wrap)
}
