//! GGUF (GGML Universal File) format parser — llama.cpp's quantized
//! tensor container.
//!
//! Spec: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>
//!
//! This module owns the wire-format types ([`Magic`],
//! [`VersionedMagic`], [`ValueType`], [`Value`], [`TensorInfo`],
//! [`Content`]) and the byte-level reader. The Tensor-construction
//! layer (`tensor()`, `tensor_from_mmap()`, `write()` of QTensor
//! payloads) lives in `fuel-core/src/quantized/gguf_file.rs` since
//! those steps need a `Device` and read/write `QTensor` data.

use std::collections::HashMap;
use std::io::{Read, Seek, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use fuel_ir::{Context, Error, GgmlDType, Result, Shape, bail};

pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Magic {
    Gguf,
}

impl TryFrom<u32> for Magic {
    type Error = Error;
    fn try_from(value: u32) -> Result<Self> {
        let magic = match value {
            0x46554747 | 0x47475546 => Self::Gguf,
            _ => bail!("unknown magic 0x{value:08x}"),
        };
        Ok(magic)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgufV1,
    GgufV2,
    GgufV3,
}

impl VersionedMagic {
    pub fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let magic = reader.read_u32::<LittleEndian>()?;
        let magic = Magic::try_from(magic)?;
        let version = reader.read_u32::<LittleEndian>()?;
        let versioned_magic = match (magic, version) {
            (Magic::Gguf, 1) => Self::GgufV1,
            (Magic::Gguf, 2) => Self::GgufV2,
            (Magic::Gguf, 3) => Self::GgufV3,
            _ => bail!("gguf: unsupported magic/version {magic:?}/{version}"),
        };
        Ok(versioned_magic)
    }
}

/// Metadata describing a single tensor's location and layout in the
/// file.
#[derive(Debug)]
pub struct TensorInfo {
    pub ggml_dtype: GgmlDType,
    pub shape: Shape,
    pub offset: u64,
}

pub fn read_string<R: Read>(reader: &mut R, magic: &VersionedMagic) -> Result<String> {
    let len = match magic {
        VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as usize,
        VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
            reader.read_u64::<LittleEndian>()? as usize
        }
    };
    let mut v = vec![0u8; len];
    reader.read_exact(&mut v)?;
    // GGUF strings are supposed to be non-null terminated but in practice this happens.
    while let Some(0) = v.last() {
        v.pop();
    }
    // GGUF strings are utf8 encoded but there are cases that don't seem to be valid.
    Ok(String::from_utf8_lossy(&v).into_owned())
}

pub fn write_string<W: Write>(w: &mut W, s: &str) -> Result<()> {
    let bytes = s.as_bytes();
    w.write_u64::<LittleEndian>(bytes.len() as u64)?;
    w.write_all(bytes)?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    /// 8-bit unsigned integer.
    U8,
    /// 8-bit signed integer.
    I8,
    /// 16-bit unsigned little-endian integer.
    U16,
    /// 16-bit signed little-endian integer.
    I16,
    /// 32-bit unsigned little-endian integer.
    U32,
    /// 32-bit signed little-endian integer.
    I32,
    /// 64-bit unsigned little-endian integer.
    U64,
    /// 64-bit signed little-endian integer.
    I64,
    /// 32-bit IEEE754 floating point number.
    F32,
    /// 64-bit IEEE754 floating point number.
    F64,
    /// Boolean — 1 byte; `0` is false, `1` is true.
    Bool,
    /// UTF-8 non-null-terminated string with length prepended.
    String,
    /// Array of other values; length and value-type prepended. Arrays
    /// can be nested.
    Array,
}

impl ValueType {
    pub fn from_u32(v: u32) -> Result<Self> {
        let v = match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            v => bail!("unrecognized value-type {v:#08x}"),
        };
        Ok(v)
    }

    pub fn to_u32(self) -> u32 {
        match self {
            Self::U8 => 0,
            Self::I8 => 1,
            Self::U16 => 2,
            Self::I16 => 3,
            Self::U32 => 4,
            Self::I32 => 5,
            Self::F32 => 6,
            Self::Bool => 7,
            Self::String => 8,
            Self::Array => 9,
            Self::U64 => 10,
            Self::I64 => 11,
            Self::F64 => 12,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F32(_) => ValueType::F32,
            Self::F64(_) => ValueType::F64,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::Array(_) => ValueType::Array,
        }
    }

    pub fn to_u8(&self) -> Result<u8> {
        match self {
            Self::U8(v) => Ok(*v),
            v => bail!("not a u8 {v:?}"),
        }
    }

    pub fn to_i8(&self) -> Result<i8> {
        match self {
            Self::I8(v) => Ok(*v),
            v => bail!("not a i8 {v:?}"),
        }
    }

    pub fn to_u16(&self) -> Result<u16> {
        match self {
            Self::U16(v) => Ok(*v),
            v => bail!("not a u16 {v:?}"),
        }
    }

    pub fn to_i16(&self) -> Result<i16> {
        match self {
            Self::I16(v) => Ok(*v),
            v => bail!("not a i16 {v:?}"),
        }
    }

    pub fn to_u32(&self) -> Result<u32> {
        match self {
            Self::U32(v) => Ok(*v),
            v => bail!("not a u32 {v:?}"),
        }
    }

    pub fn to_i32(&self) -> Result<i32> {
        match self {
            Self::I32(v) => Ok(*v),
            v => bail!("not a i32 {v:?}"),
        }
    }

    /// Auto-upcasts smaller unsigned integers and `bool` to `u64`.
    pub fn to_u64(&self) -> Result<u64> {
        match self {
            Self::U64(v) => Ok(*v),
            Self::U8(v) => Ok(*v as u64),
            Self::U16(v) => Ok(*v as u64),
            Self::U32(v) => Ok(*v as u64),
            Self::Bool(v) => Ok(*v as u64),
            v => bail!("not a u64 or upcastable to u64 {v:?}"),
        }
    }

    pub fn to_i64(&self) -> Result<i64> {
        match self {
            Self::I64(v) => Ok(*v),
            v => bail!("not a i64 {v:?}"),
        }
    }

    pub fn to_f32(&self) -> Result<f32> {
        match self {
            Self::F32(v) => Ok(*v),
            v => bail!("not a f32 {v:?}"),
        }
    }

    pub fn to_f64(&self) -> Result<f64> {
        match self {
            Self::F64(v) => Ok(*v),
            v => bail!("not a f64 {v:?}"),
        }
    }

    pub fn to_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            v => bail!("not a bool {v:?}"),
        }
    }

    pub fn to_vec(&self) -> Result<&Vec<Value>> {
        match self {
            Self::Array(v) => Ok(v),
            v => bail!("not a vec {v:?}"),
        }
    }

    #[allow(clippy::inherent_to_string_shadow_display)]
    pub fn to_string(&self) -> Result<&String> {
        match self {
            Self::String(v) => Ok(v),
            v => bail!("not a string {v:?}"),
        }
    }

    pub fn read<R: Read>(
        reader: &mut R,
        value_type: ValueType,
        magic: &VersionedMagic,
    ) -> Result<Self> {
        let v = match value_type {
            ValueType::U8 => Self::U8(reader.read_u8()?),
            ValueType::I8 => Self::I8(reader.read_i8()?),
            ValueType::U16 => Self::U16(reader.read_u16::<LittleEndian>()?),
            ValueType::I16 => Self::I16(reader.read_i16::<LittleEndian>()?),
            ValueType::U32 => Self::U32(reader.read_u32::<LittleEndian>()?),
            ValueType::I32 => Self::I32(reader.read_i32::<LittleEndian>()?),
            ValueType::U64 => Self::U64(reader.read_u64::<LittleEndian>()?),
            ValueType::I64 => Self::I64(reader.read_i64::<LittleEndian>()?),
            ValueType::F32 => Self::F32(reader.read_f32::<LittleEndian>()?),
            ValueType::F64 => Self::F64(reader.read_f64::<LittleEndian>()?),
            ValueType::Bool => match reader.read_u8()? {
                0 => Self::Bool(false),
                1 => Self::Bool(true),
                b => bail!("unexpected bool value {b}"),
            },
            ValueType::String => Self::String(read_string(reader, magic)?),
            ValueType::Array => {
                let value_type = reader.read_u32::<LittleEndian>()?;
                let value_type = ValueType::from_u32(value_type)?;
                let len = match magic {
                    VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as usize,
                    VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                        reader.read_u64::<LittleEndian>()? as usize
                    }
                };
                let mut vs = Vec::with_capacity(len);
                for _ in 0..len {
                    vs.push(Value::read(reader, value_type, magic)?);
                }
                Self::Array(vs)
            }
        };
        Ok(v)
    }

    pub fn write<W: Write>(&self, w: &mut W) -> Result<()> {
        match self {
            &Self::U8(v) => w.write_u8(v)?,
            &Self::I8(v) => w.write_i8(v)?,
            &Self::U16(v) => w.write_u16::<LittleEndian>(v)?,
            &Self::I16(v) => w.write_i16::<LittleEndian>(v)?,
            &Self::U32(v) => w.write_u32::<LittleEndian>(v)?,
            &Self::I32(v) => w.write_i32::<LittleEndian>(v)?,
            &Self::U64(v) => w.write_u64::<LittleEndian>(v)?,
            &Self::I64(v) => w.write_i64::<LittleEndian>(v)?,
            &Self::F32(v) => w.write_f32::<LittleEndian>(v)?,
            &Self::F64(v) => w.write_f64::<LittleEndian>(v)?,
            &Self::Bool(v) => w.write_u8(u8::from(v))?,
            Self::String(v) => write_string(w, v.as_str())?,
            Self::Array(v) => {
                // The `Value` type does not enforce that all the
                // values in an Array have the same type.
                let value_type = if v.is_empty() {
                    ValueType::U32
                } else {
                    let value_type: std::collections::HashSet<_> =
                        v.iter().map(|elem| elem.value_type()).collect();
                    if value_type.len() != 1 {
                        bail!("multiple value-types in the same array {value_type:?}")
                    }
                    value_type.into_iter().next().context("empty value_type")?
                };
                w.write_u32::<LittleEndian>(value_type.to_u32())?;
                w.write_u64::<LittleEndian>(v.len() as u64)?;
                for elem in v.iter() {
                    elem.write(w)?;
                }
            }
        }
        Ok(())
    }
}

/// Parsed GGUF header — magic, KV metadata, tensor table, and the
/// computed `tensor_data_offset`.
#[derive(Debug)]
pub struct Content {
    pub magic: VersionedMagic,
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    pub tensor_data_offset: u64,
}

impl Content {
    /// Parse a GGUF header from `reader`. After this call, `reader`'s
    /// stream position points to the start of the tensor data
    /// section (post-padding).
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let magic = VersionedMagic::read(reader)?;

        let tensor_count = match magic {
            VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as usize,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                reader.read_u64::<LittleEndian>()? as usize
            }
        };
        let metadata_kv_count = match magic {
            VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as usize,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                reader.read_u64::<LittleEndian>()? as usize
            }
        };

        let mut metadata = HashMap::new();
        for _idx in 0..metadata_kv_count {
            let key = read_string(reader, &magic)?;
            let value_type = reader.read_u32::<LittleEndian>()?;
            let value_type = ValueType::from_u32(value_type)?;
            let value = Value::read(reader, value_type, &magic)?;
            metadata.insert(key, value);
        }
        let mut tensor_infos = HashMap::new();
        for _idx in 0..tensor_count {
            let tensor_name = read_string(reader, &magic)?;
            let n_dimensions = reader.read_u32::<LittleEndian>()?;

            let mut dimensions: Vec<usize> = match magic {
                VersionedMagic::GgufV1 => {
                    let mut dimensions = vec![0; n_dimensions as usize];
                    reader.read_u32_into::<LittleEndian>(&mut dimensions)?;
                    dimensions.into_iter().map(|c| c as usize).collect()
                }
                VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                    let mut dimensions = vec![0; n_dimensions as usize];
                    reader.read_u64_into::<LittleEndian>(&mut dimensions)?;
                    dimensions.into_iter().map(|c| c as usize).collect()
                }
            };

            dimensions.reverse();
            let ggml_dtype = reader.read_u32::<LittleEndian>()?;
            let ggml_dtype = GgmlDType::from_u32(ggml_dtype)?;
            let offset = reader.read_u64::<LittleEndian>()?;
            tensor_infos.insert(
                tensor_name,
                TensorInfo {
                    shape: Shape::from(dimensions),
                    offset,
                    ggml_dtype,
                },
            );
        }
        let position = reader.stream_position()?;
        let alignment = match metadata.get("general.alignment") {
            Some(Value::U8(v)) => *v as u64,
            Some(Value::U16(v)) => *v as u64,
            Some(Value::U32(v)) => *v as u64,
            Some(Value::I8(v)) if *v >= 0 => *v as u64,
            Some(Value::I16(v)) if *v >= 0 => *v as u64,
            Some(Value::I32(v)) if *v >= 0 => *v as u64,
            _ => DEFAULT_ALIGNMENT,
        };
        let tensor_data_offset = position.div_ceil(alignment) * alignment;
        Ok(Self {
            magic,
            metadata,
            tensor_infos,
            tensor_data_offset,
        })
    }
}
