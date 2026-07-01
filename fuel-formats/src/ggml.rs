//! GGML legacy tensor format parser.
//!
//! Wire layout: `[magic][optional version][HParams][Vocab][tensor]*`
//! repeated until EOF. Each tensor is described by `(n_dims, name_len,
//! ggml_dtype, dims..., name, [pad to 32], raw_bytes)`. Dimensions
//! are stored in reverse order on disk and reversed back on read.
//!
//! This module owns the wire-format types ([`Magic`],
//! [`VersionedMagic`], [`HParams`], [`Vocab`], [`RawTensor`]) and the
//! readers that produce them from any `impl Read + Seek`. Promotion
//! of [`RawTensor`] to a typed `QTensor` lives downstream
//! (`fuel-core/src/quantized/ggml_file.rs::qtensor_from_ggml`).
//!
//! Reference: <https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.cpp#L505>

use std::io::{Read, Seek, SeekFrom};

use byteorder::{LittleEndian, ReadBytesExt};
use fuel_ir::{Error, GgmlDType, Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Magic {
    Ggjt,
    Ggla,
    Ggmf,
    Ggml,
    Ggsn,
}

impl TryFrom<u32> for Magic {
    type Error = Error;
    fn try_from(value: u32) -> Result<Self> {
        let magic = match value {
            0x67676a74 => Self::Ggjt,
            0x67676c61 => Self::Ggla,
            0x67676d66 => Self::Ggmf,
            0x67676d6c => Self::Ggml,
            0x6767736e => Self::Ggsn,
            _ => bail!("unknown magic {value:08x}"),
        };
        Ok(magic)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgmlUnversioned,
    GgmfV1,
    GgjtV1,
    GgjtV2,
    GgjtV3,
}

impl VersionedMagic {
    pub fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let magic = reader.read_u32::<LittleEndian>()?;
        let magic = Magic::try_from(magic)?;
        if magic == Magic::Ggml {
            return Ok(Self::GgmlUnversioned);
        }
        let version = reader.read_u32::<LittleEndian>()?;
        let versioned_magic = match (magic, version) {
            (Magic::Ggmf, 1) => Self::GgmfV1,
            (Magic::Ggjt, 1) => Self::GgjtV1,
            (Magic::Ggjt, 2) => Self::GgjtV2,
            (Magic::Ggjt, 3) => Self::GgjtV3,
            _ => bail!("ggml: unsupported magic/version {magic:?}/{version}"),
        };
        Ok(versioned_magic)
    }

    /// True for versions that 32-byte-align tensor payloads on disk.
    pub fn align32(&self) -> bool {
        match self {
            Self::GgmlUnversioned | Self::GgmfV1 => false,
            Self::GgjtV1 | Self::GgjtV2 | Self::GgjtV3 => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HParams {
    pub n_vocab: u32,
    pub n_embd: u32,
    pub n_mult: u32,
    pub n_head: u32,
    pub n_layer: u32,
    pub n_rot: u32,
    pub ftype: u32,
}

impl HParams {
    pub fn read<R: Read>(reader: &mut R) -> Result<Self> {
        Ok(Self {
            n_vocab: reader.read_u32::<LittleEndian>()?,
            n_embd: reader.read_u32::<LittleEndian>()?,
            n_mult: reader.read_u32::<LittleEndian>()?,
            n_head: reader.read_u32::<LittleEndian>()?,
            n_layer: reader.read_u32::<LittleEndian>()?,
            n_rot: reader.read_u32::<LittleEndian>()?,
            ftype: reader.read_u32::<LittleEndian>()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Vocab {
    pub token_score_pairs: Vec<(Vec<u8>, f32)>,
}

impl Vocab {
    /// Read `n_vocab` `(token_bytes, score)` pairs from the stream.
    pub fn read<R: Read>(reader: &mut R, n_vocab: usize) -> Result<Self> {
        // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.cpp#L556
        let mut token_score_pairs = Vec::with_capacity(n_vocab);
        for _ in 0..n_vocab {
            let len = reader.read_u32::<LittleEndian>()? as usize;
            let mut word = vec![0u8; len];
            reader.read_exact(&mut word)?;
            let score = reader.read_f32::<LittleEndian>()?;
            token_score_pairs.push((word, score));
        }
        Ok(Self { token_score_pairs })
    }
}

/// Parsed file header — magic, hyperparameters, and vocabulary.
///
/// Tensors follow the header on disk and are read individually via
/// [`read_one_raw_tensor`].
pub struct Header {
    pub magic: VersionedMagic,
    pub hparams: HParams,
    pub vocab: Vocab,
}

impl Header {
    /// Read magic + HParams + Vocab from the stream's current position.
    /// Caller is responsible for positioning the reader at the file
    /// start beforehand if needed.
    pub fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let magic = VersionedMagic::read(reader)?;
        let hparams = HParams::read(reader)?;
        let vocab = Vocab::read(reader, hparams.n_vocab as usize)?;
        Ok(Self {
            magic,
            hparams,
            vocab,
        })
    }
}

/// Parsed-but-not-yet-tensorified GGML tensor entry.
///
/// `data` is the raw on-disk byte payload, length matching
/// `dims.iter().product() / dtype.block_size() * dtype.type_size()`.
/// Promotion to a typed quantized tensor is the caller's job.
pub struct RawTensor {
    pub name: String,
    pub dtype: GgmlDType,
    pub dims: Vec<usize>,
    pub data: Vec<u8>,
}

/// Read one tensor entry from the stream. The reader must support
/// `Seek` because `align32` versions pad payloads to 32-byte
/// boundaries.
pub fn read_one_raw_tensor<R: Read + Seek>(
    reader: &mut R,
    magic: VersionedMagic,
) -> Result<RawTensor> {
    let n_dims = reader.read_u32::<LittleEndian>()?;
    let name_len = reader.read_u32::<LittleEndian>()?;
    let ggml_dtype = reader.read_u32::<LittleEndian>()?;
    let ggml_dtype = GgmlDType::from_u32(ggml_dtype)?;
    let mut dims = vec![0u32; n_dims as usize];
    reader.read_u32_into::<LittleEndian>(&mut dims)?;
    // Dimensions are stored in reverse order on disk, see e.g.:
    // https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/convert.py#L969
    dims.reverse();
    let mut name = vec![0u8; name_len as usize];
    reader.read_exact(&mut name)?;
    let name = String::from_utf8_lossy(&name).into_owned();

    if magic.align32() {
        let pos = reader.stream_position()?;
        reader.seek(SeekFrom::Current(((32 - pos % 32) % 32) as i64))?;
    }
    let dims = dims.iter().map(|&u| u as usize).collect::<Vec<_>>();
    let tensor_elems = dims.iter().product::<usize>();
    let size_in_bytes = tensor_elems * ggml_dtype.type_size() / ggml_dtype.block_size();
    let mut data = vec![0u8; size_in_bytes];
    reader.read_exact(&mut data)?;
    Ok(RawTensor {
        name,
        dtype: ggml_dtype,
        dims,
        data,
    })
}
