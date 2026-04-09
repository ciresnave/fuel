//! Varbuilder for Loading gguf files
//!
//! VarBuilder is a utility to store quantized tensors from a [GGUF model file](https://huggingface.co/docs/hub/gguf).
//! These tensors can be loaded from disk using `from_gguf` or from an in-memory
//! buffer using `from_gguf_buffer`.

use candle::quantized::QTensor;
use candle::{Device, Result, Shape};
use std::sync::Arc;

/// A variable-builder for loading quantized tensors from GGUF model files.
///
/// Wraps an in-memory map of [`QTensor`] values keyed by their fully-qualified
/// GGUF tensor names and exposes a hierarchical path-prefixing API that mirrors
/// [`candle_nn::VarBuilder`].
///
/// Tensors are read and validated eagerly when created via
/// [`from_gguf`](VarBuilder::from_gguf) or
/// [`from_gguf_buffer`](VarBuilder::from_gguf_buffer).
// VarBuilder specialized for QTensors
#[derive(Clone)]
pub struct VarBuilder {
    data: Arc<std::collections::HashMap<String, Arc<QTensor>>>,
    path: Vec<String>,
    device: Device,
}

impl VarBuilder {
    /// Opens a GGUF file at `p` and loads all tensors into memory.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened, the GGUF header is
    /// malformed, or any individual tensor fails to load onto `device`.
    pub fn from_gguf<P: AsRef<std::path::Path>>(p: P, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(p)?;
        let content = candle::quantized::gguf_file::Content::read(&mut file)?;
        let mut data = std::collections::HashMap::new();
        for tensor_name in content.tensor_infos.keys() {
            let tensor = content.tensor(&mut file, tensor_name, device)?;
            data.insert(tensor_name.to_string(), Arc::new(tensor));
        }
        Ok(Self {
            data: Arc::new(data),
            path: Vec::new(),
            device: device.clone(),
        })
    }

    /// Reads a GGUF model from an in-memory `buffer` and loads all tensors.
    ///
    /// Useful when the model file has already been read into memory (e.g. in
    /// a WASM environment where direct file I/O is unavailable).
    pub fn from_gguf_buffer(buffer: &[u8], device: &Device) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(buffer);
        let content = candle::quantized::gguf_file::Content::read(&mut cursor)?;
        let mut data = std::collections::HashMap::new();
        for tensor_name in content.tensor_infos.keys() {
            let tensor = content.tensor(&mut cursor, tensor_name, device)?;
            data.insert(tensor_name.to_string(), Arc::new(tensor));
        }
        Ok(Self {
            data: Arc::new(data),
            path: Vec::new(),
            device: device.clone(),
        })
    }

    /// Returns a child [`VarBuilder`] with `s` appended to the tensor name prefix.
    ///
    /// This mirrors the `pp` ("push path") convention from [`candle_nn::VarBuilder`].
    /// Successive calls accumulate dot-separated path segments, e.g.
    /// `vb.pp("model").pp("layers").pp("0")` resolves tensors under `"model.layers.0."`.
    pub fn pp<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            data: self.data.clone(),
            path,
            device: self.device.clone(),
        }
    }

    fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    /// Retrieves a quantized tensor by name, verifying it matches shape `s`.
    ///
    /// The full tensor key is formed by joining the current path prefix with
    /// `name` using a dot separator.
    ///
    /// # Errors
    /// Returns an error if the tensor is not found or its shape does not match `s`.
    pub fn get<S: Into<Shape>>(&self, s: S, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);
        match self.data.get(&path) {
            None => {
                candle::bail!("cannot find tensor {path}")
            }
            Some(qtensor) => {
                let shape = s.into();
                if qtensor.shape() != &shape {
                    candle::bail!(
                        "shape mismatch for {name}, got {:?}, expected {shape:?}",
                        qtensor.shape()
                    )
                }
                Ok(qtensor.clone())
            }
        }
    }

    /// Retrieves a quantized tensor by name without shape validation.
    ///
    /// Prefer [`get`](VarBuilder::get) when the expected shape is known.
    pub fn get_no_shape(&self, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);
        match self.data.get(&path) {
            None => {
                candle::bail!("cannot find tensor {name}")
            }
            Some(qtensor) => Ok(qtensor.clone()),
        }
    }

    /// Returns the device on which tensors are loaded.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns `true` if a tensor with the fully-qualified `key` exists in this builder.
    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }
}
