//! GGUF file format — fuel-core's metadata view.
//!
//! Wire-format parsing (magic, KV metadata, tensor-info table, value
//! decode/encode) lives in [`fuel_formats::gguf`]. This file re-exports
//! that surface and wraps the parsed header in a fuel-core [`Content`].
//!
//! B6 removed the eager half: `Content::tensor`, `Content::tensor_from_mmap`,
//! and the free `write` function all built or consumed the eager `QTensor`.
//! Reading tensor *bytes* needs none of that — take the `TensorInfo` from
//! [`Content::tensor_infos`] plus [`Content::tensor_data_offset`] and slice the
//! backing mmap yourself. That is what the lazy loaders already do; see
//! `fuel-core/src/lazy_quantized_llama.rs` and its siblings.

use std::collections::HashMap;
use std::io::{Read, Seek};

use crate::Result;

pub use fuel_formats::gguf::{
    DEFAULT_ALIGNMENT, TensorInfo, Value, ValueType, VersionedMagic, read_string, write_string,
};

/// Parsed GGUF header: metadata, the tensor-info table, and the byte
/// offset at which tensor data begins.
///
/// Field shape mirrors [`fuel_formats::gguf::Content`].
#[derive(Debug)]
pub struct Content {
    pub magic: VersionedMagic,
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    pub tensor_data_offset: u64,
}

impl Content {
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let parsed = fuel_formats::gguf::Content::read(reader)?;
        Ok(Self {
            magic: parsed.magic,
            metadata: parsed.metadata,
            tensor_infos: parsed.tensor_infos,
            tensor_data_offset: parsed.tensor_data_offset,
        })
    }
}
