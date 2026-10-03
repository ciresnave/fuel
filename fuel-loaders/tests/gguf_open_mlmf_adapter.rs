// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pins `Content::open` (the MLMF-backed constructor) against
//! `Content::read` (the original, untouched byte-stream parser) on
//! synthetic GGUF files built by hand, byte-for-byte.
//!
//! Wire-format reference: `fuel-formats/tests/transport_independence.rs`'s
//! `gguf_minimal_header_parses_from_in_memory_cursor` for the header, and
//! `mlmf-gguf/src/tensors.rs`'s own `info()` test helper for the
//! tensor-info record layout (both read directly at `origin/main` while
//! drafting this file, not inferred).

use std::io::Cursor;

use byteorder::{LittleEndian, WriteBytesExt};
use fuel_loaders::quantized::gguf_file::Content;

const GGUF_MAGIC: u32 = 0x4655_4747; // b"GGUF" little-endian
const DEFAULT_ALIGNMENT: u64 = 32;

/// One tensor-info record: name (u64 len + bytes), n_dims (u32), dims (u64
/// each, FILE declaration order -- callers reverse for fuel's convention),
/// ggml type code (u32), offset (u64, relative to tensor_data_offset).
fn tensor_info_record(name: &str, dims: &[u64], ggml_code: u32, offset: u64) -> Vec<u8> {
    let mut b = Vec::new();
    b.write_u64::<LittleEndian>(name.len() as u64).unwrap();
    b.extend_from_slice(name.as_bytes());
    b.write_u32::<LittleEndian>(dims.len() as u32).unwrap();
    for d in dims {
        b.write_u64::<LittleEndian>(*d).unwrap();
    }
    b.write_u32::<LittleEndian>(ggml_code).unwrap();
    b.write_u64::<LittleEndian>(offset).unwrap();
    b
}

/// One metadata entry: key (u64 len + bytes), value_type (u32: 8 = String
/// per `fuel_formats::gguf::ValueType::to_u32`), value (u64 len + raw
/// bytes -- NOT necessarily valid UTF-8, per the Bytes-lossy test below).
fn metadata_string_entry(key: &str, raw_value: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    b.write_u64::<LittleEndian>(key.len() as u64).unwrap();
    b.extend_from_slice(key.as_bytes());
    b.write_u32::<LittleEndian>(8).unwrap(); // ValueType::String
    b.write_u64::<LittleEndian>(raw_value.len() as u64).unwrap();
    b.extend_from_slice(raw_value);
    b
}

/// Assembles a complete, byte-exact GGUF v3 file: header, metadata block,
/// tensor-info table, zero-padding up to the next 32-byte boundary, then
/// `tensor_data`. Returns the whole buffer.
fn build_gguf(
    metadata_entries: &[Vec<u8>],
    tensor_infos: &[Vec<u8>],
    tensor_data: &[u8],
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap(); // version
    buf.write_u64::<LittleEndian>(tensor_infos.len() as u64)
        .unwrap();
    buf.write_u64::<LittleEndian>(metadata_entries.len() as u64)
        .unwrap();
    for m in metadata_entries {
        buf.extend_from_slice(m);
    }
    for t in tensor_infos {
        buf.extend_from_slice(t);
    }
    let position = buf.len() as u64;
    let data_offset = position.div_ceil(DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT;
    buf.resize(data_offset as usize, 0);
    buf.extend_from_slice(tensor_data);
    buf
}

/// POSITIVE CONTROL: a well-formed single-F32-tensor file, read through
/// both constructors, must produce equal `Content`s. This is the exact
/// comparison the design review asked for before trusting `open` on
/// anything else.
#[test]
fn gguf_open_matches_read_for_an_f32_tensor() {
    let tensor_data = 0.0f32.to_le_bytes().repeat(4); // a 2x2 f32 tensor, all zero
    let gguf = build_gguf(
        &[],
        &[tensor_info_record("t", &[2, 2], /* F32 */ 0, 0)],
        &tensor_data,
    );

    let mut cursor = Cursor::new(gguf.clone());
    let via_read = Content::read(&mut cursor).expect("Content::read on a well-formed file");

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("t.gguf");
    std::fs::write(&path, &gguf).expect("write synthetic gguf");
    let via_open = Content::open(&path).expect("Content::open on the same bytes");

    assert_eq!(via_read.magic, via_open.magic);
    assert_eq!(via_read.tensor_data_offset, via_open.tensor_data_offset);
    assert_eq!(via_read.metadata.len(), via_open.metadata.len());
    assert_eq!(via_read.tensor_infos.len(), via_open.tensor_infos.len());
    let (r, o) = (&via_read.tensor_infos["t"], &via_open.tensor_infos["t"]);
    assert_eq!(r.shape.dims(), o.shape.dims(), "dimension order must agree");
    assert_eq!(r.offset, o.offset);
    assert_eq!(r.ggml_dtype, o.ggml_dtype);
}

/// A tensor declaring ggml code 24 (I8) -- a type `GgmlType::encoding()`
/// resolves to `Encoding::Dense(DType::I8)`, which fuel's narrower
/// `GgmlDType` has no variant for. `Content::read` already fails this file
/// with a typed Err (`GgmlDType::from_u32` has no arm for 24); `open` must
/// fail the same way, never panic, never silently drop the tensor.
#[test]
fn gguf_open_returns_typed_err_for_an_i8_tensor() {
    let gguf = build_gguf(
        &[],
        &[tensor_info_record("weird", &[4], /* I8 */ 24, 0)],
        &[0i8; 4].iter().map(|&b| b as u8).collect::<Vec<_>>(),
    );

    let mut cursor = Cursor::new(gguf.clone());
    let read_err = Content::read(&mut cursor).expect_err("fuel's own parser already rejects I8");

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("weird.gguf");
    std::fs::write(&path, &gguf).expect("write synthetic gguf");
    let open_err = Content::open(&path).expect_err("open must reject I8 too, not panic");

    // Both are typed Errs (not a panic -- the call above would have
    // aborted the test process otherwise), and both name the same root
    // cause: an unknown/unsupported ggml dtype code.
    let read_msg = read_err.to_string();
    let open_msg = open_err.to_string();
    assert!(
        read_msg.contains("dtype") || read_msg.contains("24"),
        "fuel's own error should name the dtype or code: {read_msg}"
    );
    assert!(
        open_msg.contains("dtype") || open_msg.contains("24") || open_msg.contains("weird"),
        "open's error should name the dtype, code, or tensor: {open_msg}"
    );
}

/// GAP pin: a metadata string whose declared bytes are not valid UTF-8.
/// `Content::read` already converts it LOSSILY (`read_string`'s own
/// `String::from_utf8_lossy`); `open` must match that exact lossy output,
/// not fail, not preserve the raw bytes -- the fix (adding `Value::Bytes`)
/// is deliberately deferred to its own breaking-wave PR. This test is
/// meant to go red the day that fix lands, on purpose.
#[test]
fn gguf_open_non_utf8_metadata_string_is_lossy_like_read() {
    let raw = [0x66u8, 0x6f, 0xff, 0x6f]; // "fo\xFFo" -- 0xFF is never valid UTF-8
    let gguf = build_gguf(&[metadata_string_entry("k", &raw)], &[], &[]);

    let mut cursor = Cursor::new(gguf.clone());
    let via_read = Content::read(&mut cursor).expect("read tolerates non-UTF-8 via lossy decode");

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("lossy.gguf");
    std::fs::write(&path, &gguf).expect("write synthetic gguf");
    let via_open = Content::open(&path).expect("open must match read's lossy behavior");

    // `Value` derives no `PartialEq` -- match the String variant by hand.
    let expected = String::from_utf8_lossy(&raw).into_owned();
    match via_read.metadata.get("k") {
        Some(fuel_formats::gguf::Value::String(s)) => assert_eq!(*s, expected),
        other => panic!("Content::read: expected Value::String({expected:?}), got {other:?}"),
    }
    match via_open.metadata.get("k") {
        Some(fuel_formats::gguf::Value::String(s)) => assert_eq!(
            *s, expected,
            "open's lossy mapping must match read's exactly -- this test goes \
             red on purpose the day Value::Bytes lands and this arm stops being lossy"
        ),
        other => panic!("Content::open: expected Value::String({expected:?}), got {other:?}"),
    }
}
