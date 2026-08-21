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
use std::io::{Read, Seek, SeekFrom, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use fuel_ir::{Context, Error, GgmlDType, Result, Shape, bail};

pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Maximum GGUF array nesting depth. Arrays can nest, and the reader recurses
/// per level; an untrusted file with unbounded nesting is a stack-overflow
/// abort. ggml itself never nests deeply — this bound is generous.
const MAX_ARRAY_DEPTH: usize = 64;

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
    // Read up to `len` bytes WITHOUT pre-allocating `len`: a `.gguf` is
    // untrusted and can declare a length far larger than it contains, so
    // `vec![0u8; len]` is a `len`-sized allocation an attacker controls (abort
    // from a 4/8-byte field). `take` + `read_to_end` grows `v` to the ACTUAL
    // bytes available; if fewer than `len` arrive the declared length is a lie.
    let mut v = Vec::new();
    let read = reader.by_ref().take(len as u64).read_to_end(&mut v)?;
    if read != len {
        bail!("gguf: declared string length {len} exceeds available data ({read} bytes read)");
    }
    // ⚠️ KNOWN LIMITATION (GAP-205 — DEFERRED to `mlmf-gguf`, not a bug to fix
    // here). A GGUF string is a length-prefixed BYTE ARRAY, but this function is
    // LOSSY twice: it (a) strips a RUN of trailing NULs and (b) replaces invalid
    // UTF-8 with U+FFFD via `from_utf8_lossy`. It is used for metadata keys,
    // string values, tokens, merges AND TENSOR NAMES — so a tensor name that is
    // NUL-terminated or non-UTF-8 no longer byte-matches the file, and a lookup
    // by the original name silently fails. The fix is to keep names as raw bytes
    // (MLMF's replacement adds a `Bytes(Vec<u8>)` value for exactly this). It is
    // an API-semantics change deliberately deferred to the `mlmf-gguf` crate that
    // replaces this file, NOT made on this retirement-path file — and the harm on
    // real tensor names is as yet UNMEASURED, which an API break would need first.
    while let Some(0) = v.last() {
        v.pop();
    }
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
        Self::read_bounded(reader, value_type, magic, 0)
    }

    /// Depth-tracked reader. `depth` is the current array-nesting level; the
    /// public [`Value::read`] enters at 0. Arrays recurse at `depth + 1` and
    /// are rejected past [`MAX_ARRAY_DEPTH`] so a hostile file cannot overflow
    /// the stack.
    fn read_bounded<R: Read>(
        reader: &mut R,
        value_type: ValueType,
        magic: &VersionedMagic,
        depth: usize,
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
                if depth >= MAX_ARRAY_DEPTH {
                    bail!("gguf: array nesting exceeds the limit of {MAX_ARRAY_DEPTH}");
                }
                let value_type = reader.read_u32::<LittleEndian>()?;
                let value_type = ValueType::from_u32(value_type)?;
                let len = match magic {
                    VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as usize,
                    VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                        reader.read_u64::<LittleEndian>()? as usize
                    }
                };
                // Do NOT `Vec::with_capacity(len)` — `len` is untrusted and can
                // be far larger than the file holds. `Vec::new()` grows to the
                // ACTUAL element count; a too-large `len` runs out of file and
                // the inner read errors, bounding memory to real data.
                let mut vs = Vec::new();
                for _ in 0..len {
                    vs.push(Value::read_bounded(reader, value_type, magic, depth + 1)?);
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
    /// ⚠️ KNOWN LIMITATION (GAP-205 — DEFERRED to `mlmf-gguf`): a `HashMap` drops
    /// the file's declared order and SILENTLY LAST-WINS on a duplicate metadata
    /// key or tensor name — a malformed or hostile file with a repeated key
    /// keeps only the last value, with no error. MLMF's replacement uses a
    /// Vec-in-declared-order plus an index, with a duplicate tensor name made
    /// fatal; that ordering/dup fix is an API change deferred to the replacement
    /// crate, not made on this retirement-path file.
    pub metadata: HashMap<String, Value>,
    /// See the note on the `metadata` field — the same `HashMap` last-wins
    /// caveat applies to duplicate tensor names.
    pub tensor_infos: HashMap<String, TensorInfo>,
    pub tensor_data_offset: u64,
}

impl Content {
    /// Parse a GGUF header from `reader`. After this call, `reader`'s
    /// stream position points to the start of the tensor data
    /// section (post-padding).
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        // Total stream length — used to reject declared lengths/counts that
        // exceed what the file can hold (each declared item needs >= 1 byte on
        // disk). Preserve the reader's starting position rather than forcing 0,
        // so this keeps reading from wherever the caller left it.
        let start = reader.stream_position()?;
        let file_len = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(start))?;
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
            // Each dimension is 4 (v1) or 8 (v2/v3) bytes on disk, so a count
            // exceeding the bytes left in the file is malformed — reject before
            // `vec![0; n]` (n_dimensions is a u32: up to 34 GB from a 4-byte field).
            let remaining = file_len.saturating_sub(reader.stream_position()?);
            if n_dimensions as u64 > remaining {
                bail!(
                    "gguf: tensor dimension count {n_dimensions} exceeds the {remaining} bytes left in the file"
                );
            }

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
        // `general.alignment` is UNTRUSTED file metadata. Reject 0 — the
        // `div_ceil(0)` below is an integer divide-by-zero that aborts the
        // process — and non-powers-of-two — ggml pads with a `GGML_PAD`
        // bitmask, so `div_ceil` would compute a DIFFERENT `tensor_data_offset`
        // than every other reader (pos 11, align 3 → 12 vs 13), silently
        // reading the wrong bytes. `is_power_of_two()` is false for both 0 and
        // any non-power-of-two, so this one check covers both.
        if !alignment.is_power_of_two() {
            bail!("gguf: general.alignment must be a non-zero power of two, got {alignment}");
        }
        let tensor_data_offset = position.div_ceil(alignment) * alignment;
        Ok(Self {
            magic,
            metadata,
            tensor_infos,
            tensor_data_offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal valid GGUF v3 header (0 tensors) with an optional
    /// `general.alignment` U32 metadata value. Exercises the alignment
    /// validation path with hand-crafted, potentially-malformed input —
    /// the whole point being that a `.gguf` is UNTRUSTED user data.
    fn gguf_with_alignment(align: Option<u32>) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        buf.write_u32::<LittleEndian>(0x4655_4747).unwrap(); // "GGUF" magic
        buf.write_u32::<LittleEndian>(3).unwrap(); // version 3
        buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count = 0
        buf.write_u64::<LittleEndian>(align.is_some() as u64)
            .unwrap(); // metadata_kv_count
        if let Some(a) = align {
            write_string(&mut buf, "general.alignment").unwrap();
            buf.write_u32::<LittleEndian>(ValueType::U32.to_u32())
                .unwrap();
            buf.write_u32::<LittleEndian>(a).unwrap();
        }
        buf
    }

    /// `general.alignment == 0` must be a typed error, NOT a divide-by-zero
    /// abort. `position.div_ceil(0)` panics ("attempt to divide by zero"),
    /// reachable from any untrusted `.gguf` — a never-panic violation on the
    /// most exposed path Fuel has. Born-red: without the check this test
    /// PANICS inside `Content::read` before it can return.
    #[test]
    fn alignment_zero_is_rejected_not_a_panic() {
        let buf = gguf_with_alignment(Some(0));
        let err = Content::read(&mut Cursor::new(buf))
            .expect_err("alignment 0 must be a typed error, not a panic");
        let msg = format!("{err}");
        assert!(
            msg.contains("alignment") && msg.contains('0'),
            "error must name the offending value: {msg}"
        );
    }

    /// A non-power-of-two alignment must be REJECTED. ggml pads with a
    /// `GGML_PAD` bitmask, so `div_ceil` and the bitmask DISAGREE — position
    /// 11 with alignment 3 gives 12 via `div_ceil` and 13 via the bitmask —
    /// so we would compute a different `tensor_data_offset` than every other
    /// reader, SILENTLY. Born-red: without the check this returns `Ok` with a
    /// wrong offset and `expect_err` fails.
    #[test]
    fn alignment_non_power_of_two_is_rejected() {
        let buf = gguf_with_alignment(Some(3));
        let err = Content::read(&mut Cursor::new(buf))
            .expect_err("non-power-of-two alignment must be a typed error");
        assert!(format!("{err}").contains('3'), "error must name the value");
    }

    /// POSITIVE CONTROL: a valid power-of-two alignment parses, and the data
    /// offset is aligned up from the end of the header.
    #[test]
    fn alignment_power_of_two_is_accepted() {
        let buf = gguf_with_alignment(Some(64));
        let header_len = buf.len() as u64;
        let content =
            Content::read(&mut Cursor::new(buf)).expect("valid power-of-two alignment parses");
        assert_eq!(
            content.tensor_data_offset % 64,
            0,
            "offset must be 64-aligned"
        );
        assert!(content.tensor_data_offset >= header_len);
    }

    /// POSITIVE CONTROL: absent `general.alignment` falls back to the 32-byte
    /// default and parses — the check must not reject the common case.
    #[test]
    fn absent_alignment_uses_default() {
        let content = Content::read(&mut Cursor::new(gguf_with_alignment(None)))
            .expect("default alignment parses");
        assert_eq!(content.tensor_data_offset % DEFAULT_ALIGNMENT, 0);
    }

    // ---- hardening: unbounded allocations from declared lengths ----

    /// Build a v3 GGUF with a single metadata KV whose raw value bytes are
    /// supplied verbatim, so a test can hand-craft a malformed value.
    fn gguf_with_kv(key: &str, value_type: ValueType, value_bytes: &[u8]) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        buf.write_u32::<LittleEndian>(0x4655_4747).unwrap(); // "GGUF"
        buf.write_u32::<LittleEndian>(3).unwrap();
        buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // metadata_kv_count
        write_string(&mut buf, key).unwrap();
        buf.write_u32::<LittleEndian>(value_type.to_u32()).unwrap();
        buf.extend_from_slice(value_bytes);
        buf
    }

    fn u64le(v: u64) -> Vec<u8> {
        let mut b = Vec::new();
        b.write_u64::<LittleEndian>(v).unwrap();
        b
    }

    /// A string value declaring `u64::MAX` bytes with none present must be a
    /// typed error, not a `vec![0u8; u64::MAX]` allocation. Born-red: without
    /// the fix that allocation is a capacity-overflow PANIC (or, at a length
    /// between RAM and isize::MAX, an uncatchable OOM abort).
    #[test]
    fn string_declared_length_beyond_data_is_rejected() {
        let buf = gguf_with_kv("k", ValueType::String, &u64le(u64::MAX));
        let err = Content::read(&mut Cursor::new(buf))
            .expect_err("an over-long declared string length must be a typed error");
        let msg = format!("{err}").to_lowercase();
        assert!(
            msg.contains("string") || msg.contains("length"),
            "names the problem: {msg}"
        );
    }

    /// An array declaring `u64::MAX` elements with none present must be a typed
    /// error, not `Vec::with_capacity(u64::MAX)`. Born-red: without the fix that
    /// preallocation is a capacity-overflow PANIC.
    #[test]
    fn array_declared_length_beyond_data_is_rejected() {
        let mut val = Vec::new();
        val.write_u32::<LittleEndian>(ValueType::U8.to_u32())
            .unwrap(); // element type
        val.extend_from_slice(&u64le(u64::MAX)); // element count
        let buf = gguf_with_kv("k", ValueType::Array, &val);
        Content::read(&mut Cursor::new(buf))
            .expect_err("an over-long declared array length must be a typed error");
    }

    /// A tensor declaring a huge `n_dimensions` must be rejected before
    /// `vec![0; n]` — at `u32::MAX` that is 34 GB from a 4-byte field. Born-red:
    /// without the fix a large-but-not-abortive n (1e6) allocates then errors
    /// generically at read; with the fix it is a typed dimension error.
    #[test]
    fn tensor_ndimensions_beyond_file_is_rejected() {
        let mut buf: Vec<u8> = Vec::new();
        buf.write_u32::<LittleEndian>(0x4655_4747).unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap();
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count = 1
        buf.write_u64::<LittleEndian>(0).unwrap(); // metadata_kv_count = 0
        write_string(&mut buf, "t").unwrap(); // tensor name
        buf.write_u32::<LittleEndian>(1_000_000).unwrap(); // n_dimensions, far beyond the file
        let err = Content::read(&mut Cursor::new(buf))
            .expect_err("a huge n_dimensions must be a typed error before allocating");
        assert!(
            format!("{err}").to_lowercase().contains("dimension"),
            "names the field"
        );
    }

    /// Deeply-nested arrays must be bounded — unbounded recursion is a stack
    /// overflow (an uncatchable abort). Born-red: without the depth limit a
    /// 200-deep nest parses to Ok; with it, a typed nesting error.
    #[test]
    fn deeply_nested_array_is_rejected() {
        // innermost: an empty U8 array; then wrap it 200 times in single-element arrays.
        let mut val = Vec::new();
        val.write_u32::<LittleEndian>(ValueType::U8.to_u32())
            .unwrap();
        val.extend_from_slice(&u64le(0)); // empty innermost
        for _ in 0..200 {
            let mut outer = Vec::new();
            outer
                .write_u32::<LittleEndian>(ValueType::Array.to_u32())
                .unwrap();
            outer.extend_from_slice(&u64le(1)); // one element: the previous level
            outer.extend_from_slice(&val);
            val = outer;
        }
        let buf = gguf_with_kv("k", ValueType::Array, &val);
        let err = Content::read(&mut Cursor::new(buf))
            .expect_err("excessive array nesting must be a typed error");
        assert!(
            format!("{err}").to_lowercase().contains("nest"),
            "names the problem"
        );
    }
}
