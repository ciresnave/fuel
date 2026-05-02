//! Proves that every parser in `fuel-formats` works against an
//! in-memory byte buffer without touching the filesystem.
//!
//! This is the success criterion for ROADMAP Phase 7.5 work item A:
//! parsers must be transport-independent so the same code serves
//! file loads, network streaming, IPC, and shared-memory consumers.

use std::io::{Cursor, Write};

use byteorder::{LittleEndian, WriteBytesExt};

#[test]
fn imatrix_parses_from_in_memory_cursor() {
    // Build an imatrix payload by hand and parse it via Cursor<&[u8]>.
    let mut buf = Vec::new();
    buf.write_i32::<LittleEndian>(1).unwrap();
    buf.write_i32::<LittleEndian>("layer".len() as i32).unwrap();
    buf.write_all(b"layer").unwrap();
    buf.write_i32::<LittleEndian>(2).unwrap(); // ncall
    buf.write_i32::<LittleEndian>(3).unwrap(); // nval
    buf.write_f32::<LittleEndian>(2.0).unwrap();
    buf.write_f32::<LittleEndian>(4.0).unwrap();
    buf.write_f32::<LittleEndian>(6.0).unwrap();

    let parsed = fuel_formats::imatrix::parse_bytes(&buf).unwrap();
    // ncall=2 normalization → values divided by 2.
    assert_eq!(parsed["layer"], vec![1.0, 2.0, 3.0]);
}

#[test]
fn ggml_header_parses_from_in_memory_cursor() {
    use fuel_formats::ggml::{Header, VersionedMagic};

    // Magic + (no version, GgmlUnversioned) + HParams + empty Vocab.
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(0x67676d6c).unwrap(); // 'l','m','g','g' → Magic::Ggml → GgmlUnversioned
    // HParams: 7 × u32
    for v in [32_000u32, 4096, 256, 32, 32, 128, 0] {
        buf.write_u32::<LittleEndian>(v).unwrap();
    }
    // Vocab: zero entries because we set n_vocab=32_000... that would loop. Use a smaller value.

    // Rewrite with n_vocab=0 so the test can finish.
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(0x67676d6c).unwrap();
    for v in [0u32, 4096, 256, 32, 32, 128, 0] {
        buf.write_u32::<LittleEndian>(v).unwrap();
    }
    // Vocab is empty (n_vocab=0).

    let mut cursor = Cursor::new(buf);
    let header = Header::read(&mut cursor).unwrap();
    assert_eq!(header.magic, VersionedMagic::GgmlUnversioned);
    assert_eq!(header.hparams.n_embd, 4096);
    assert_eq!(header.vocab.token_score_pairs.len(), 0);
}

#[test]
fn gguf_minimal_header_parses_from_in_memory_cursor() {
    use fuel_formats::gguf::{Content, VersionedMagic};

    // Build a minimal GGUF v3 header with zero metadata and zero
    // tensors — enough to exercise the parser without any QTensor /
    // Device coupling.
    let mut buf = Vec::new();
    // Magic: 0x46554747 (`GGUF` little-endian) ; Version: 3
    buf.write_u32::<LittleEndian>(0x46554747).unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap();
    // tensor_count (u64 for v2/v3): 0
    buf.write_u64::<LittleEndian>(0).unwrap();
    // metadata_kv_count (u64 for v2/v3): 0
    buf.write_u64::<LittleEndian>(0).unwrap();

    let mut cursor = Cursor::new(buf);
    let content = Content::read(&mut cursor).unwrap();
    assert_eq!(content.magic, VersionedMagic::GgufV3);
    assert_eq!(content.tensor_infos.len(), 0);
    assert_eq!(content.metadata.len(), 0);
}

#[test]
fn safetensors_parses_from_in_memory_buffer() {
    use fuel_formats::safetensors::SafeTensors;

    // Minimal safetensors payload: header length (u64 LE) + header
    // JSON, then 16 bytes of f32 zeros for a 2x2 tensor.
    let header = b"{\"t\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}       ";
    let mut buf = Vec::new();
    buf.write_u64::<LittleEndian>(header.len() as u64).unwrap();
    buf.extend_from_slice(header);
    buf.extend_from_slice(&[0u8; 16]);

    let st = SafeTensors::deserialize(&buf).unwrap();
    let names: Vec<_> = st.names().into_iter().collect();
    assert_eq!(names, vec!["t"]);
    let view = st.tensor("t").unwrap();
    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(view.data().len(), 16);
}

#[test]
fn pickle_stack_runs_against_in_memory_cursor() {
    use fuel_formats::pickle::{Object, Stack};
    use std::io::BufReader;

    // Encode a tiny pickle stream by hand: PROTO 2, NEWTRUE, STOP.
    // This is the smallest valid pickle that finalizes to a real object.
    let bytes: &[u8] = &[
        0x80, 0x02, // PROTO 2
        0x88, // NEWTRUE
        b'.', // STOP
    ];
    let mut reader = BufReader::new(Cursor::new(bytes));
    let mut stack = Stack::empty();
    stack.read_loop(&mut reader).unwrap();
    let obj = stack.finalize().unwrap();
    assert_eq!(obj, Object::Bool(true));
}
