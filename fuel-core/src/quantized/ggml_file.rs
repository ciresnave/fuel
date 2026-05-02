//! GGML file format — Tensor-construction layer.
//!
//! Wire-format parsing (magic detection, HParams, Vocab, raw-tensor
//! reading) lives in [`fuel_formats::ggml`]. This file holds the
//! `Device`-aware promotion: turning each [`fuel_formats::ggml::RawTensor`]
//! into a [`super::QTensor`].

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use crate::{Device, Result};

pub use fuel_formats::ggml::{HParams, Header, RawTensor, Vocab, VersionedMagic};

use super::{GgmlDType, QTensor};

/// Creates a [`QTensor`] from a raw GGML tensor.
pub fn qtensor_from_ggml(
    ggml_dtype: GgmlDType,
    raw_data: &[u8],
    dims: Vec<usize>,
    device: &Device,
) -> Result<QTensor> {
    let tensor_elems = dims.iter().product::<usize>();
    let block_size = ggml_dtype.block_size();
    if tensor_elems % block_size != 0 {
        crate::bail!(
            "the number of elements {tensor_elems} is not divisible by the block size {block_size}"
        )
    }
    let size_in_bytes = tensor_elems / block_size * ggml_dtype.type_size();
    let bytes = &raw_data[..size_in_bytes];
    let storage = super::load_quantized(std::borrow::Cow::Borrowed(bytes), device, ggml_dtype)?;
    QTensor::new(storage, dims)
}

pub struct Content {
    pub magic: VersionedMagic,
    pub hparams: HParams,
    pub vocab: Vocab,
    pub tensors: HashMap<String, QTensor>,
    pub device: Device,
}

impl Content {
    pub fn read<R: Read + Seek>(reader: &mut R, device: &Device) -> Result<Content> {
        let last_position = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(0))?;
        let header = Header::read(reader)?;
        let mut tensors = HashMap::new();
        while reader.stream_position()? != last_position {
            let raw = fuel_formats::ggml::read_one_raw_tensor(reader, header.magic)?;
            let name = raw.name.clone();
            let tensor = qtensor_from_ggml(raw.dtype, &raw.data, raw.dims, device)
                .map_err(|e| crate::Error::msg(format!("Error creating tensor {name}: {e}")))?;
            tensors.insert(name, tensor);
        }
        Ok(Self {
            magic: header.magic,
            hparams: header.hparams,
            vocab: header.vocab,
            tensors,
            device: device.clone(),
        })
    }

    pub fn remove(&mut self, name: &str) -> Result<QTensor> {
        match self.tensors.remove(name) {
            None => crate::bail!("cannot find tensor with name '{name}'"),
            Some(tensor) => Ok(tensor),
        }
    }
}
