// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pins `Content::open` (the MLMF-backed constructor, now the only GGUF reader) on synthetic GGUF
//! files built by hand, byte for byte, against values written down here. It used to compare
//! `open` with `Content::read`, fuel's own byte-stream parser; that parser was deleted in MLMF
//! slice D after `tests/gguf_corpus_equivalence.rs` showed the two agreeing on 30 of 31 real
//! files (the 31st is GGUF v1, which MLMF refuses on purpose), so the expectations below are
//! the ones the old comparison established.
//!
//! Wire-format reference: `fuel-formats/tests/transport_independence.rs`'s
//! `gguf_minimal_header_parses_from_in_memory_cursor` for the header, and
//! `mlmf-gguf/src/tensors.rs`'s own `info()` test helper for the
//! tensor-info record layout (both read directly at `origin/main` while
//! drafting this file, not inferred).

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

/// Write `gguf` to a temp file and open it.
fn open_bytes(gguf: &[u8], name: &str) -> Content {
    try_open_bytes(gguf, name).unwrap_or_else(|e| panic!("{name}: open must succeed, got {e}"))
}

fn try_open_bytes(gguf: &[u8], name: &str) -> fuel_ir::error::Result<Content> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(name);
    std::fs::write(&path, gguf).expect("write synthetic gguf");
    Content::open(&path)
}

/// Shared fixture: a well-formed file with THREE tensors spanning both MLMF encoding branches
/// (F32 and F16 are `Encoding::Dense`; Q4_0 is `Encoding::Blocked`, exercising the `BlockSpec.code`
/// passthrough separately from the Dense-DType hand-mapping) plus TWO real metadata entries (a
/// scalar U32 and a String).
fn three_tensor_fixture() -> Content {
    // Q4_0: 32 elements/block, 18 bytes/block -- one whole block, dims=[32].
    let q4_0_data = vec![0u8; 18];
    let f32_data = 0.0f32.to_le_bytes().repeat(4); // 2x2
    let f16_data = vec![0u8; 2 * 3]; // 3 f16 elements

    // Tensor-data offsets are RELATIVE to tensor_data_offset and must not overlap: lay them out
    // back-to-back, each one 32-byte-aligned (GGUF's own per-tensor alignment rule).
    let (f32_off, f16_off, q4_0_off) = (0u64, 32u64, 64u64);
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
    open_bytes(&gguf, "t.gguf")
}

#[test]
fn gguf_open_reads_three_tensors() {
    use fuel_ir::GgmlDType;
    let c = three_tensor_fixture();
    assert_eq!(c.tensor_data_offset % 32, 0);
    assert_eq!(c.tensor_infos.len(), 3);
    for (name, dims, dtype, offset) in [
        ("weight.f32", vec![2usize, 2], GgmlDType::F32, 0u64),
        ("weight.f16", vec![3], GgmlDType::F16, 32),
        ("weight.q4_0", vec![32], GgmlDType::Q4_0, 64),
    ] {
        let t = &c.tensor_infos[name];
        assert_eq!(t.shape.dims(), dims.as_slice(), "{name}: dimension order");
        assert_eq!(t.offset, offset, "{name}: offset");
        assert_eq!(t.ggml_dtype, dtype, "{name}: dtype");
    }
}

#[test]
fn gguf_open_reads_real_metadata() {
    let c = three_tensor_fixture();
    assert_eq!(c.metadata.len(), 2);
    // `Value` derives no `PartialEq` -- match each variant by hand, for BOTH metadata keys.
    match c.metadata.get("general.some_count") {
        Some(fuel_formats::gguf::Value::U32(v)) => assert_eq!(*v, 42),
        other => panic!("general.some_count: expected Value::U32(42), got {other:?}"),
    }
    match c.metadata.get("general.name") {
        Some(fuel_formats::gguf::Value::String(s)) => assert_eq!(s, "test-model"),
        other => panic!("general.name: expected Value::String, got {other:?}"),
    }
}

/// GAP-211: a tensor whose ggml type code fuel has no variant for (24 = I8, and the whole IQ*/TQ*
/// family) must NOT stop the file from opening -- a metadata-only consumer (a `chat_template`
/// reader) would otherwise be held hostage to the tensor type table. `open` must (1) succeed,
/// (2) keep the supported tensor, (3) record the unsupported one by name with its raw code, and
/// (4) turn a LOAD of that tensor into a typed error naming the tensor and the code -- never the
/// generic "missing tensor", never silence. A name the file does not declare keeps the plain
/// "missing tensor" message (positive control: the two messages must differ).
#[test]
fn gguf_unsupported_tensor_type_opens_and_fails_by_name_on_load() {
    for (label, code) in [("i8", 24u32), ("q1_0", 41u32), ("q2_0", 42u32)] {
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
        let content = open_bytes(&gguf, "unsupported.gguf");
        assert!(
            content.metadata.contains_key("general.name"),
            "{label}: metadata must survive"
        );
        assert!(
            content.tensor_infos.contains_key("good"),
            "{label}: the supported tensor is listed"
        );
        assert!(
            !content.tensor_infos.contains_key("weird"),
            "{label}: the unsupported tensor must not be listed as loadable"
        );
        assert_eq!(
            content.unsupported_tensors.get("weird"),
            Some(&code),
            "{label}: the unsupported tensor is recorded with its raw code"
        );
        let msg = content.missing_tensor_error("weird").to_string();
        assert!(
            msg.contains("weird") && msg.contains(&code.to_string()),
            "{label}: load error must name the tensor and the code: {msg}"
        );
        let absent = content.missing_tensor_error("never_declared").to_string();
        assert!(
            absent.contains("missing tensor") && !absent.contains("type code"),
            "{label}: an undeclared name keeps the plain message: {absent}"
        );
    }
}

/// GAP-211 B1/B2: every IQ type (ggml codes 16..=23 and 29) is a SUPPORTED type, so `open` lists
/// them as ordinary tensors with the right dtype and byte offsets, and nothing lands in
/// `unsupported_tensors`. (The sibling test uses types that stay unsupported; together they pin
/// the boundary.)
#[test]
fn gguf_iq_tensors_open_as_supported_types() {
    use fuel_ir::GgmlDType;
    let cases: [(&str, u32, GgmlDType); 13] = [
        ("iq2_xxs", 16, GgmlDType::Iq2Xxs),
        ("iq2_xs", 17, GgmlDType::Iq2Xs),
        ("iq3_xxs", 18, GgmlDType::Iq3Xxs),
        ("iq1_s", 19, GgmlDType::Iq1S),
        ("iq4_nl", 20, GgmlDType::Iq4Nl),
        ("iq3_s", 21, GgmlDType::Iq3S),
        ("iq2_s", 22, GgmlDType::Iq2S),
        ("iq4_xs", 23, GgmlDType::Iq4Xs),
        ("iq1_m", 29, GgmlDType::Iq1M),
        ("tq1_0", 34, GgmlDType::Tq1_0),
        ("tq2_0", 35, GgmlDType::Tq2_0),
        ("mxfp4", 39, GgmlDType::Mxfp4),
        ("nvfp4", 40, GgmlDType::Nvfp4),
    ];
    // One block of each, laid out back to back.
    let mut offset = 0u64;
    let mut records = Vec::new();
    for (name, code, dtype) in &cases {
        records.push(tensor_info_record(
            name,
            &[dtype.block_size() as u64],
            *code,
            offset,
        ));
        offset += dtype.type_size() as u64;
    }
    let data = vec![0u8; offset as usize];
    let c = open_bytes(&build_gguf(&[], &records, &data), "iq.gguf");
    assert!(
        c.unsupported_tensors.is_empty(),
        "{:?}",
        c.unsupported_tensors
    );
    let mut want = 0u64;
    for (name, _, dtype) in &cases {
        assert_eq!(c.tensor_infos[*name].ggml_dtype, *dtype, "{name}");
        assert_eq!(c.tensor_infos[*name].offset, want, "{name}");
        want += dtype.type_size() as u64;
    }
}

/// A metadata string whose declared bytes are not valid UTF-8 comes back LOSSILY converted
/// (`String::from_utf8_lossy`), as the deleted `fuel-formats` reader did: not an error, and not the
/// raw bytes. The fix (a real `Value::Bytes` variant) is deliberately deferred to its own breaking
/// PR; this test is meant to go red the day it lands, on purpose.
#[test]
fn gguf_open_non_utf8_metadata_string_is_lossy() {
    let raw = [0x66u8, 0x6f, 0xff, 0x6f]; // 0xFF is never valid UTF-8
    let c = open_bytes(
        &build_gguf(&[metadata_string_entry("k", &raw)], &[], &[]),
        "lossy.gguf",
    );
    let expected = String::from_utf8_lossy(&raw).into_owned();
    match c.metadata.get("k") {
        Some(fuel_formats::gguf::Value::String(s)) => assert_eq!(
            *s, expected,
            "the lossy mapping must be exact; this test goes red on purpose the day Value::Bytes lands"
        ),
        other => panic!("expected Value::String({expected:?}), got {other:?}"),
    }
}

/// `general.alignment` declared as `I32(64)`: MLMF's `GgufMetadata::alignment()` accepts only
/// `U32` and falls back to 32 for every other type, which would put every tensor's data at the
/// wrong offset SILENTLY. `open` must either refuse the file or resolve the 64-byte alignment: a
/// `tensor_data_offset` that is not a multiple of 64 is the silent-wrong-data finding (adversarial
/// review of #302).
#[test]
fn gguf_open_rejects_alignment_type_mismatch_rather_than_silently_differ() {
    let gguf = build_gguf(
        &[metadata_i32_entry("general.alignment", 64)],
        &[tensor_info_record("t", &[4], /* F32 */ 0, 0)],
        &0.0f32.to_le_bytes().repeat(4),
    );
    if let Ok(c) = try_open_bytes(&gguf, "align.gguf") {
        assert_eq!(
            c.tensor_data_offset % 64,
            0,
            "open resolved a tensor_data_offset that ignores the declared I32 alignment"
        );
    }
}

/// GGUF v1 (32-bit counts) is refused on purpose by `mlmf-gguf`, and fuel keeps no second parser
/// (MLMF slice D, PM ruling 2026-10-24): a typed error that names the version and the remedy, not
/// a panic and not a mis-parse. The real v1 file in the corpus is checked in
/// `gguf_corpus_equivalence.rs`; this is the synthetic arm that runs everywhere.
#[test]
fn gguf_v1_is_refused_with_a_message_naming_the_version_and_the_fix() {
    let mut gguf = build_gguf(&[], &[], &[]);
    gguf[4..8].copy_from_slice(&1u32.to_le_bytes()); // the version field
    let err = try_open_bytes(&gguf, "v1.gguf")
        .expect_err("v1 must be refused")
        .to_string();
    assert!(
        err.contains("GGUF v1"),
        "the message names the version: {err}"
    );
    assert!(
        err.contains("re-export"),
        "the message names the remedy: {err}"
    );
}
