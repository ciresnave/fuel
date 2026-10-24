// SPDX-License-Identifier: MIT OR Apache-2.0
//! MLMF move, slice B (`docs/design-notes/2026-10-24-mlmf-move-survey.md`): the safetensors
//! reader. This test is the equivalence gate: it was written and run against the OLD reader
//! (the upstream `safetensors` crate behind `MmapedSafetensors`) BEFORE the new one was built,
//! and it must stay green after.
//!
//! The reference is the upstream `safetensors::SafeTensors::deserialize` applied directly to the
//! file's bytes, which shares no code with fuel's reader. For every tensor of every file the
//! reader exposes it must give the same name set, dtype, shape and BYTES.
//!
//! Files: hand-built ones (every dtype safetensors defines, a zero-length tensor, two shards
//! that both define one name, where the last shard wins), plus the real checkpoints named by
//! `FUEL_QWEN3_AWQ_DIR` (`Orion-zhen/Qwen3-0.6B-AWQ`) and `FUEL_QWEN35_DIR` (`Qwen/Qwen3.5-0.8B`),
//! each a `Hardware::Hub` skip when unset.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use fuel_loaders::safetensors::{BufferedSafetensors, MmapedSafetensors};
use fuel_test_support::hardware::{Hardware, Missing, skip};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

/// `(dtype, shape, bytes)` per tensor name, from the reference reader.
fn reference(bytes: &[u8]) -> BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)> {
    let st = SafeTensors::deserialize(bytes).expect("reference reader");
    st.tensors()
        .into_iter()
        .map(|(n, v)| (n, (v.dtype(), v.shape().to_vec(), v.data().to_vec())))
        .collect()
}

fn same(
    label: &str,
    got: &BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
    want: &BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
) {
    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        want.keys().collect::<Vec<_>>(),
        "{label}: tensor name sets differ"
    );
    for (name, w) in want {
        let g = &got[name];
        assert_eq!(g.0, w.0, "{label}: dtype of {name}");
        assert_eq!(g.1, w.1, "{label}: shape of {name}");
        assert!(g.2 == w.2, "{label}: bytes of {name} differ");
    }
}

/// SAFETY (all four mapping calls in this file go through these two helpers): every file mapped
/// here is a test file written to a temp dir, or a read-only checkpoint, that nothing modifies or
/// truncates while the mapping is alive.
fn open_one(p: &Path) -> fuel_ir::error::Result<MmapedSafetensors> {
    // SAFETY: see above.
    unsafe { MmapedSafetensors::new(p) }
}

fn open_many(paths: &[&Path]) -> fuel_ir::error::Result<MmapedSafetensors> {
    // SAFETY: see above.
    unsafe { MmapedSafetensors::multi(paths) }
}

fn via_mmap(paths: &[&Path]) -> BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)> {
    let st = if let [one] = paths {
        open_one(one)
    } else {
        open_many(paths)
    }
    .expect("mmap reader");
    let listed: BTreeMap<_, _> = st
        .tensors()
        .into_iter()
        .map(|(n, v)| (n, (v.dtype(), v.shape().to_vec(), v.data().to_vec())))
        .collect();
    // `get` must agree with `tensors` for every name.
    for (name, (dtype, shape, data)) in &listed {
        let v = st.get(name).expect("get");
        assert_eq!(
            (v.dtype(), v.shape().to_vec(), v.data().to_vec()),
            (*dtype, shape.clone(), data.clone()),
            "get({name}) vs tensors()"
        );
    }
    listed
}

fn serialize(tensors: Vec<(&str, Dtype, Vec<usize>, Vec<u8>)>) -> Vec<u8> {
    let owned: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = tensors
        .into_iter()
        .map(|(n, d, s, b)| (n.to_string(), d, s, b))
        .collect();
    let views: Vec<(String, TensorView<'_>)> = owned
        .iter()
        .map(|(n, d, s, b)| (n.clone(), TensorView::new(*d, s.clone(), b).expect("view")))
        .collect();
    safetensors::tensor::serialize(views, None).expect("serialize")
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

/// One tensor of each dtype safetensors defines that `mlmf-safetensors` knows, a zero-length
/// tensor, and a scalar.
fn dtype_zoo() -> Vec<u8> {
    let mut t = Vec::new();
    let dtypes = [
        ("f64", Dtype::F64, 8),
        ("f32", Dtype::F32, 4),
        ("f16", Dtype::F16, 2),
        ("bf16", Dtype::BF16, 2),
        ("f8e4m3", Dtype::F8_E4M3, 1),
        ("f8e5m2", Dtype::F8_E5M2, 1),
        ("i64", Dtype::I64, 8),
        ("i32", Dtype::I32, 4),
        ("i16", Dtype::I16, 2),
        ("i8", Dtype::I8, 1),
        ("u64", Dtype::U64, 8),
        ("u32", Dtype::U32, 4),
        ("u16", Dtype::U16, 2),
        ("u8", Dtype::U8, 1),
        ("bool", Dtype::BOOL, 1),
    ];
    for (i, (name, dt, width)) in dtypes.into_iter().enumerate() {
        t.push((name, dt, vec![3, 5], pattern(15 * width, i as u8)));
    }
    t.push(("empty", Dtype::F32, vec![0, 7], vec![]));
    t.push(("scalar", Dtype::F32, vec![], pattern(4, 99)));
    serialize(t)
}

#[test]
fn the_mmap_reader_matches_the_reference_on_every_dtype() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zoo.safetensors");
    let bytes = dtype_zoo();
    std::fs::write(&path, &bytes).unwrap();
    let want = reference(&bytes);
    assert_eq!(
        want.len(),
        17,
        "positive control: 15 dtypes + empty + scalar"
    );
    same("mmap", &via_mmap(&[&path]), &want);
}

#[test]
fn the_buffered_reader_matches_the_reference() {
    let bytes = dtype_zoo();
    let want = reference(&bytes);
    let st = BufferedSafetensors::new(bytes).expect("buffered reader");
    let got: BTreeMap<_, _> = st
        .tensors()
        .into_iter()
        .map(|(n, v)| (n, (v.dtype(), v.shape().to_vec(), v.data().to_vec())))
        .collect();
    same("buffered", &got, &want);
    for (name, (dtype, shape, data)) in &got {
        let v = st.get(name).expect("get");
        assert_eq!(
            (v.dtype(), v.shape().to_vec(), v.data().to_vec()),
            (*dtype, shape.clone(), data.clone())
        );
    }
}

/// Two shards define `shared`; the LAST one wins (the documented routing rule), and a name only
/// in the first shard is still reachable.
#[test]
fn multi_shard_routing_is_last_shard_wins() {
    let dir = tempfile::tempdir().unwrap();
    let a = serialize(vec![
        ("only_a", Dtype::U8, vec![4], pattern(4, 1)),
        ("shared", Dtype::U8, vec![2], vec![1, 1]),
    ]);
    let b = serialize(vec![
        ("only_b", Dtype::F32, vec![2], pattern(8, 2)),
        ("shared", Dtype::U8, vec![3], vec![9, 9, 9]),
    ]);
    let (pa, pb) = (
        dir.path().join("a.safetensors"),
        dir.path().join("b.safetensors"),
    );
    std::fs::write(&pa, &a).unwrap();
    std::fs::write(&pb, &b).unwrap();
    let got = via_mmap(&[&pa, &pb]);
    assert_eq!(
        got.keys().cloned().collect::<Vec<_>>(),
        ["only_a", "only_b", "shared"]
    );
    assert_eq!(got["shared"].2, vec![9, 9, 9], "the last shard wins");
    let mut want = reference(&a);
    want.extend(reference(&b));
    same("multi", &got, &want);
}

#[test]
fn a_missing_name_and_a_truncated_file_are_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zoo.safetensors");
    let bytes = dtype_zoo();
    std::fs::write(&path, &bytes).unwrap();
    let st = open_one(&path).unwrap();
    assert!(st.get("nope").is_err());
    let cut = dir.path().join("cut.safetensors");
    std::fs::write(&cut, &bytes[..bytes.len() - 10]).unwrap();
    assert!(open_one(&cut).is_err(), "a truncated file must be refused");
    let junk = dir.path().join("junk.safetensors");
    std::fs::write(&junk, b"not a safetensors file at all").unwrap();
    assert!(open_one(&junk).is_err());
}

fn real_checkpoint(var: &str, what: &str) -> Option<PathBuf> {
    let Some(dir) = std::env::var_os(var).map(PathBuf::from) else {
        skip(
            Hardware::Hub,
            Missing::device(format!("{var} is not set ({what})")),
        );
        return None;
    };
    let file = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.extension().is_some_and(|x| x == "safetensors"));
    if file.is_none() {
        skip(
            Hardware::Hub,
            Missing::device(format!("{} has no .safetensors file", dir.display())),
        );
    }
    file
}

fn compare_real(file: &Path, label: &str) {
    let bytes = std::fs::read(file).expect("read checkpoint");
    let want = reference(&bytes);
    println!("{label}: {} tensors, {} bytes", want.len(), bytes.len());
    assert!(
        want.len() > 100,
        "positive control: a real checkpoint has many tensors"
    );
    drop(bytes);
    same(label, &via_mmap(&[file]), &want);
}

#[test]
fn the_mmap_reader_matches_the_reference_on_the_real_awq_checkpoint() {
    if let Some(f) = real_checkpoint("FUEL_QWEN3_AWQ_DIR", "Orion-zhen/Qwen3-0.6B-AWQ") {
        compare_real(&f, "Qwen3-0.6B-AWQ");
    }
}

#[test]
fn the_mmap_reader_matches_the_reference_on_the_real_qwen35_checkpoint() {
    if let Some(f) = real_checkpoint("FUEL_QWEN35_DIR", "Qwen/Qwen3.5-0.8B") {
        compare_real(&f, "Qwen3.5-0.8B");
    }
}

/// Sabotage sibling: the comparison can fail. One flipped byte in one tensor of a copy of the
/// zoo must be reported by `same`.
#[test]
#[should_panic(expected = "bytes of")]
fn the_comparison_notices_a_flipped_byte() {
    let mut bytes = dtype_zoo();
    let want = reference(&bytes);
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF; // the final byte belongs to the `scalar` tensor's data
    same("sabotage", &reference(&bytes), &want);
}
