//! GGUF file format — Tensor-construction layer.
//!
//! Wire-format parsing (magic, KV metadata, tensor-info table, value
//! decode/encode) lives in [`fuel_formats::gguf`]. This file holds
//! the `Device`-aware promotion: turning a [`TensorInfo`] entry plus
//! a backing reader (or mmap slice) into a [`super::QTensor`], and
//! the inverse `write` for serializing tensors back to disk.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};

use byteorder::{LittleEndian, WriteBytesExt};

use super::QTensor;
use crate::{Device, Result};

pub use fuel_formats::gguf::{
    DEFAULT_ALIGNMENT, TensorInfo, Value, ValueType, VersionedMagic, read_string, write_string,
};

/// Parsed GGUF file with `Device`-aware tensor accessors.
///
/// Field shape mirrors [`fuel_formats::gguf::Content`]; the only thing
/// fuel-core's version adds is the `tensor` / `tensor_from_mmap`
/// methods, which build [`QTensor`]s on a specific [`Device`].
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

    /// Seek-and-read the named tensor, decoding it onto `device`.
    pub fn tensor<R: Read + Seek>(
        &self,
        reader: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<QTensor> {
        let info = self
            .tensor_infos
            .get(name)
            .ok_or_else(|| crate::Error::Msg(format!("cannot find tensor info for {name}")))?;
        read_tensor_via_seek(info, reader, self.tensor_data_offset, device)
    }

    /// Zero-copy tensor lookup from an mmapped file slice.
    pub fn tensor_from_mmap(
        &self,
        tensor_data: &[u8],
        name: &str,
        device: &Device,
    ) -> Result<QTensor> {
        let info = self
            .tensor_infos
            .get(name)
            .ok_or_else(|| crate::Error::Msg(format!("cannot find tensor info for {name}")))?;
        read_tensor_from_mmap(info, tensor_data, self.tensor_data_offset, device)
    }
}

fn read_tensor_via_seek<R: Read + Seek>(
    info: &TensorInfo,
    reader: &mut R,
    tensor_data_offset: u64,
    device: &Device,
) -> Result<QTensor> {
    let tensor_elems = info.shape.elem_count();
    let block_size = info.ggml_dtype.block_size();
    if !tensor_elems.is_multiple_of(block_size) {
        crate::bail!(
            "the number of elements {tensor_elems} is not divisible by the block size {block_size}"
        )
    }
    let size_in_bytes = tensor_elems / block_size * info.ggml_dtype.type_size();
    let mut raw_data = vec![0u8; size_in_bytes];
    reader.seek(SeekFrom::Start(tensor_data_offset + info.offset))?;
    reader.read_exact(&mut raw_data)?;
    super::ggml_file::qtensor_from_ggml(
        info.ggml_dtype,
        &raw_data,
        info.shape.dims().to_vec(),
        device,
    )
}

fn read_tensor_from_mmap(
    info: &TensorInfo,
    tensor_data: &[u8],
    tensor_data_offset: u64,
    device: &Device,
) -> Result<QTensor> {
    let tensor_elems = info.shape.elem_count();
    let block_size = info.ggml_dtype.block_size();
    if !tensor_elems.is_multiple_of(block_size) {
        crate::bail!(
            "the number of elements {tensor_elems} is not divisible by the block size {block_size}"
        )
    }
    let size_in_bytes = tensor_elems / block_size * info.ggml_dtype.type_size();
    let start = (tensor_data_offset + info.offset) as usize;
    let end = start
        .checked_add(size_in_bytes)
        .ok_or_else(|| crate::Error::Msg("gguf: tensor offset overflow".into()))?;
    if end > tensor_data.len() {
        crate::bail!(
            "gguf: tensor bytes out of range (need {end}, have {})",
            tensor_data.len()
        );
    }
    super::ggml_file::qtensor_from_ggml(
        info.ggml_dtype,
        &tensor_data[start..end],
        info.shape.dims().to_vec(),
        device,
    )
}

/// Serialize `metadata` and `tensors` into `w` as a GGUF v2 file.
pub fn write<W: Seek + Write>(
    w: &mut W,
    metadata: &[(&str, &Value)],
    tensors: &[(&str, &QTensor)],
) -> Result<()> {
    w.write_u32::<LittleEndian>(0x46554747)?;
    w.write_u32::<LittleEndian>(2)?; // version 2.
    w.write_u64::<LittleEndian>(tensors.len() as u64)?;
    w.write_u64::<LittleEndian>(metadata.len() as u64)?;
    for (name, value) in metadata.iter() {
        write_string(w, name)?;
        w.write_u32::<LittleEndian>(value.value_type().to_u32())?;
        value.write(w)?;
    }
    let mut offset = 0usize;
    let mut offsets = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors.iter() {
        write_string(w, name)?;
        let dims = tensor.shape().dims();
        w.write_u32::<LittleEndian>(dims.len() as u32)?;
        for &dim in dims.iter().rev() {
            w.write_u64::<LittleEndian>(dim as u64)?;
        }
        w.write_u32::<LittleEndian>(tensor.dtype().to_u32())?;
        w.write_u64::<LittleEndian>(offset as u64)?;
        offsets.push(offset);
        let size_in_bytes = tensor.storage_size_in_bytes();
        let padding = 31 - (31 + size_in_bytes) % 32;
        offset += size_in_bytes + padding;
    }
    let pos = w.stream_position()? as usize;
    let padding = 31 - (31 + pos) % 32;
    w.write_all(&vec![0u8; padding])?;
    let tensor_start_pos = w.stream_position()? as usize;
    for (offset, (_name, tensor)) in offsets.iter().zip(tensors.iter()) {
        let pos = w.stream_position()? as usize;
        if tensor_start_pos + offset != pos {
            crate::bail!(
                "internal error, unexpected current position {tensor_start_pos} {offset} {pos}"
            )
        }
        let data = tensor.data()?;
        let size_in_bytes = data.len();
        w.write_all(&data)?;
        let padding = 31 - (31 + size_in_bytes) % 32;
        w.write_all(&vec![0u8; padding])?;
    }
    Ok(())
}
