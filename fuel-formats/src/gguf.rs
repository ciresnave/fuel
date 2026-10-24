// SPDX-License-Identifier: MIT OR Apache-2.0
//! GGUF (GGML Universal File) wire-format types and the writer — llama.cpp's quantized tensor
//! container.
//!
//! Spec: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>
//!
//! This module owns the wire-format types ([`VersionedMagic`], [`ValueType`], [`Value`],
//! [`TensorInfo`]) and the byte-level WRITER (`Value::write`, `write_string`, used by
//! `fuel-tensor-tools`). READING a GGUF is not done here: since MLMF slice D (2026-10-24) the one
//! reader is `fuel_loaders::quantized::gguf_file::Content::open`, over `mlmf-gguf`. The parser that
//! used to live here (and with it the GAP-205 panic on `general.alignment = 0`, the unbounded
//! allocations, the lossy names and the last-wins duplicate keys) is gone.

use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};
use fuel_ir::{Context, GgmlDType, Result, Shape, bail};

pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgufV1,
    GgufV2,
    GgufV3,
}

/// Metadata describing a single tensor's location and layout in the
/// file.
#[derive(Debug)]
pub struct TensorInfo {
    pub ggml_dtype: GgmlDType,
    pub shape: Shape,
    pub offset: u64,
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
