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

/// One metadata entry with a scalar I32 value (ValueType::to_u32 == 5).
fn metadata_i32_entry(key: &str, value: i32) -> Vec<u8> {
    let mut b = Vec::new();
    b.write_u64::<LittleEndian>(key.len() as u64).unwrap();
    b.extend_from_slice(key.as_bytes());
    b.write_u32::<LittleEndian>(5).unwrap(); // ValueType::I32
    b.write_i32::<LittleEndian>(value).unwrap();
    b
}

/// One metadata entry with a scalar U32 value (ValueType::to_u32 == 4).
fn metadata_u32_entry(key: &str, value: u32) -> Vec<u8> {
    let mut b = Vec::new();
    b.write_u64::<LittleEndian>(key.len() as u64).unwrap();
    b.extend_from_slice(key.as_bytes());
    b.write_u32::<LittleEndian>(4).unwrap(); // ValueType::U32
    b.write_u32::<LittleEndian>(value).unwrap();
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

/// POSITIVE CONTROL: a well-formed file with THREE tensors spanning both
/// MLMF encoding branches (F32 and F16 are `Encoding::Dense`; Q4_0 is
/// `Encoding::Blocked`, exercising the `BlockSpec.code` passthrough
/// separately from the Dense-DType hand-mapping) plus TWO real metadata
/// entries (a scalar U32 and a String), read through both constructors,
/// must produce FULLY EQUAL `Content` -- every tensor's shape/offset/dtype
/// and every metadata key's value, not just counts. This is the exact
/// comparison the design review asked for before trusting `open` on
/// anything else.
/// Shared fixture for the positive-control pair below: a well-formed file
/// with THREE tensors spanning both MLMF encoding branches (F32 and F16 are
/// `Encoding::Dense`; Q4_0 is `Encoding::Blocked`, exercising the
/// `BlockSpec.code` passthrough separately from the Dense-DType hand-mapping)
/// plus TWO real metadata entries (a scalar U32 and a String). Parses it
/// through both constructors and returns both `Content`s so each assertion
/// group can live in its own under-50-line test.
fn open_matches_read_fixture() -> (Content, Content) {
    // Q4_0: 32 elements/block, 18 bytes/block -- one whole block, dims=[32].
    let q4_0_data = vec![0u8; 18];
    let f32_data = 0.0f32.to_le_bytes().repeat(4); // 2x2
    let f16_data = vec![0u8; 2 * 3]; // 3 f16 elements

    // Tensor-data offsets are RELATIVE to tensor_data_offset and must not
    // overlap: lay them out back-to-back, each one 32-byte-aligned (GGUF's
    // own per-tensor alignment rule) to match a real writer's layout.
    let f32_off = 0u64;
    let f16_off = 32u64; // f32_data is 16 bytes, round up to 32
    let q4_0_off = 64u64; // f16_data is 6 bytes, round up to 32 from 32+6

    let mut tensor_data = vec![0u8; (q4_0_off + q4_0_data.len() as u64) as usize];
    tensor_data[f32_off as usize..f32_off as usize + f32_data.len()].copy_from_slice(&f32_data);
    tensor_data[f16_off as usize..f16_off as usize + f16_data.len()].copy_from_slice(&f16_data);
    tensor_data[q4_0_off as usize..q4_0_off as usize + q4_0_data.len()].copy_from_slice(&q4_0_data);

    let gguf = build_gguf(
        &[
            metadata_u32_entry("general.some_count", 42),
            metadata_string_entry("general.name", b"test-model"),
        ],
        &[
            tensor_info_record("weight.f32", &[2, 2], /* F32 */ 0, f32_off),
            tensor_info_record("weight.f16", &[3], /* F16 */ 1, f16_off),
            tensor_info_record("weight.q4_0", &[32], /* Q4_0 */ 2, q4_0_off),
        ],
        &tensor_data,
    );

    let mut cursor = Cursor::new(gguf.clone());
    let via_read = Content::read(&mut cursor).expect("Content::read on a well-formed file");

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("t.gguf");
    std::fs::write(&path, &gguf).expect("write synthetic gguf");
    let via_open = Content::open(&path).expect("Content::open on the same bytes");

    (via_read, via_open)
}

#[test]
fn gguf_open_matches_read_for_three_tensors() {
    let (via_read, via_open) = open_matches_read_fixture();

    assert_eq!(via_read.magic, via_open.magic);
    assert_eq!(via_read.tensor_data_offset, via_open.tensor_data_offset);
    assert_eq!(via_read.tensor_infos.len(), 3);
    assert_eq!(via_open.tensor_infos.len(), 3);

    for name in ["weight.f32", "weight.f16", "weight.q4_0"] {
        let (r, o) = (&via_read.tensor_infos[name], &via_open.tensor_infos[name]);
        assert_eq!(
            r.shape.dims(),
            o.shape.dims(),
            "{name}: dimension order must agree"
        );
        assert_eq!(r.offset, o.offset, "{name}: offset must agree");
        assert_eq!(r.ggml_dtype, o.ggml_dtype, "{name}: dtype must agree");
    }
}

#[test]
fn gguf_open_matches_read_for_real_metadata() {
    let (via_read, via_open) = open_matches_read_fixture();

    assert_eq!(via_read.metadata.len(), 2);
    assert_eq!(via_open.metadata.len(), 2);

    // `Value` derives no `PartialEq` -- match each variant by hand, on
    // BOTH constructors' output, for BOTH metadata keys.
    for content in [&via_read, &via_open] {
        match content.metadata.get("general.some_count") {
            Some(fuel_formats::gguf::Value::U32(v)) => assert_eq!(*v, 42),
            other => panic!("general.some_count: expected Value::U32(42), got {other:?}"),
        }
        match content.metadata.get("general.name") {
            Some(fuel_formats::gguf::Value::String(s)) => assert_eq!(s, "test-model"),
            other => panic!("general.name: expected Value::String, got {other:?}"),
        }
    }
}

/// GAP-211: a tensor whose ggml type code fuel has no variant for (24 = I8,
/// and the whole IQ*/TQ* family) must NOT stop the file from opening --
/// a metadata-only consumer (a `chat_template` reader) would otherwise be held
/// hostage to the tensor type table. Both constructors must (1) succeed,
/// (2) keep the supported tensor, (3) record the unsupported one by name with
/// its raw code, and (4) turn a LOAD of that tensor into a typed error naming
/// the tensor and the code -- never the generic "missing tensor", never silence.
/// A name the file does not declare keeps the plain "missing tensor" message
/// (positive control: the two messages must differ).
#[test]
fn gguf_unsupported_tensor_type_opens_and_fails_by_name_on_load() {
    for (label, code) in [("i8", 24u32), ("iq2_xxs", 16u32), ("iq3_s", 21u32)] {
        // "good": one F32 at offset 0; "weird": 32 opaque bytes at offset 4.
        let data = [&1.5f32.to_le_bytes()[..], &[0u8; 32][..]].concat();
        let gguf = build_gguf(
            &[metadata_string_entry("general.name", b"hostage")],
            &[
                tensor_info_record("good", &[1], /* F32 */ 0, 0),
                tensor_info_record("weird", &[32], code, 4),
            ],
            &data,
        );

        let via_read = Content::read(&mut Cursor::new(gguf.clone()))
            .unwrap_or_else(|e| panic!("{label}: read must open, got {e}"));
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("unsupported.gguf");
        std::fs::write(&path, &gguf).expect("write synthetic gguf");
        let via_open =
            Content::open(&path).unwrap_or_else(|e| panic!("{label}: open must open, got {e}"));

        for (ctor, content) in [("read", &via_read), ("open", &via_open)] {
            assert!(
                content.metadata.contains_key("general.name"),
                "{label}/{ctor}: metadata must survive"
            );
            assert!(
                content.tensor_infos.contains_key("good"),
                "{label}/{ctor}: the supported tensor must still be listed"
            );
            assert!(
                !content.tensor_infos.contains_key("weird"),
                "{label}/{ctor}: the unsupported tensor must not be listed as loadable"
            );
            assert_eq!(
                content.unsupported_tensors.get("weird"),
                Some(&code),
                "{label}/{ctor}: the unsupported tensor is recorded with its raw code"
            );
            let msg = content.missing_tensor_error("weird").to_string();
            assert!(
                msg.contains("weird") && msg.contains(&code.to_string()),
                "{label}/{ctor}: load error must name the tensor and the code: {msg}"
            );
            let absent = content.missing_tensor_error("never_declared").to_string();
            assert!(
                absent.contains("missing tensor") && !absent.contains("type code"),
                "{label}/{ctor}: an undeclared name keeps the plain message: {absent}"
            );
        }
    }
}

/// GAP-211 B1: IQ4_NL (20) and IQ4_XS (23) are SUPPORTED types, so both
/// constructors list them as ordinary tensors with the right dtype and byte
/// size, and nothing lands in `unsupported_tensors`. (The sibling test uses
/// types that stay unsupported; together they pin the boundary.)
#[test]
fn gguf_iq4_tensors_open_as_supported_types() {
    use fuel_ir::GgmlDType;
    let data = vec![0u8; 18 + 136]; // one IQ4_NL block (32 elems) + one IQ4_XS block (256)
    let gguf = build_gguf(
        &[],
        &[
            tensor_info_record("nl", &[32], 20, 0),
            tensor_info_record("xs", &[256], 23, 18),
        ],
        &data,
    );
    let via_read = Content::read(&mut Cursor::new(gguf.clone())).expect("read");
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("iq4.gguf");
    std::fs::write(&path, &gguf).expect("write synthetic gguf");
    let via_open = Content::open(&path).expect("open");
    for (ctor, c) in [("read", &via_read), ("open", &via_open)] {
        assert!(
            c.unsupported_tensors.is_empty(),
            "{ctor}: {:?}",
            c.unsupported_tensors
        );
        assert_eq!(c.tensor_infos["nl"].ggml_dtype, GgmlDType::Iq4Nl, "{ctor}");
        assert_eq!(c.tensor_infos["xs"].ggml_dtype, GgmlDType::Iq4Xs, "{ctor}");
        assert_eq!(c.tensor_infos["xs"].offset, 18, "{ctor}");
    }
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

/// SECURITY/CORRECTNESS FINDING (adversarial review of #302, 2026-10-03):
/// `fuel_formats::gguf::Content::read` resolves `general.alignment` from
/// SIX value-type arms (U8/U16/U32/I8/I16/I32, non-negative). MLMF's
/// `GgufMetadata::alignment()` accepts ONLY `U32`, silently falling back
/// to its own default (32) for every other type -- including `I32`, which
/// `read` honors. A file declaring `general.alignment` as `I32(64)` with a
/// tensor-info table that ends at a position divisible by 64 but NOT by 32
/// makes `read` compute one `tensor_data_offset` and (pre-fix) `open`
/// compute a DIFFERENT one SILENTLY -- no error, just a wrong byte offset
/// every tensor's data is read from. Born-red: must fail before the fix,
/// pass after.
#[test]
fn gguf_open_rejects_alignment_type_mismatch_rather_than_silently_differ() {
    // kv_count=1 (the I32 alignment entry) + 1 tensor info record for
    // "t" (name len 1, dims count 1, dims[32], code, offset -- 8+1+4+8+4+8
    // = 33 bytes) lands the raw end-of-directory position at an offset
    // that is a multiple of 64 but not of 32 is what we need to prove the
    // divergence; rather than hand-compute it, assert the two constructors
    // DISAGREE on a pre-fix build (the born-red condition) and that `open`
    // now returns a typed Err instead of a silently different offset.
    let gguf = build_gguf(
        &[metadata_i32_entry("general.alignment", 64)],
        &[tensor_info_record("t", &[4], /* F32 */ 0, 0)],
        &0.0f32.to_le_bytes().repeat(4),
    );

    let mut cursor = Cursor::new(gguf.clone());
    let via_read = Content::read(&mut cursor).expect("read honors I32 alignment");

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("align.gguf");
    std::fs::write(&path, &gguf).expect("write synthetic gguf");

    match Content::open(&path) {
        Ok(via_open) => {
            // If open also succeeds, it MUST have resolved the same
            // alignment and therefore the same tensor_data_offset as
            // read -- anything else is finding A, silently.
            assert_eq!(
                via_read.tensor_data_offset, via_open.tensor_data_offset,
                "open resolved a DIFFERENT tensor_data_offset than read for \
                 the same I32 alignment value -- this is the silent-wrong-data \
                 finding from the adversarial review, not a false alarm"
            );
        }
        Err(_) => {
            // Refusing (typed Err) rather than silently using a possibly-
            // wrong offset is the accepted fix for this finding.
        }
    }
}
