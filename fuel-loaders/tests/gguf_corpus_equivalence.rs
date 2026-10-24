// SPDX-License-Identifier: MIT OR Apache-2.0
//! MLMF move, slice D (`docs/design-notes/2026-10-24-mlmf-move-survey.md`): the GGUF reader.
//!
//! `Content::open` (MLMF-backed) is the only GGUF reader. This test is what is left of the
//! equivalence gate that justified deleting fuel's own byte-stream parser, `Content::read`. Before
//! the deletion the gate compared the two on every file of the corpus: 30 of 31 agreed on the
//! magic, the whole metadata map, the tensor-info table, the unsupported-tensor table and the
//! tensor-data offset, and the 31st is GGUF v1, which `mlmf-gguf` refuses on purpose. The 30
//! agreements were frozen as `fixtures/gguf_corpus_digests.json` (an FNV-1a hash of that digest
//! per file, produced by the OLD reader; see `fixtures/gguf_corpus_digests.PROVENANCE.md`), so the
//! comparison with the old parser survives its deletion.
//!
//! Needs `FUEL_GGUF_CORPUS` (`C:\Models\gguf-corpus`: llama.cpp vocab files, quantization samples,
//! one legacy v1 file); a `Hardware::Hub` skip when unset.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use fuel_loaders::quantized::gguf_file::Content;
use fuel_test_support::hardware::{Hardware, Missing, skip};

const GOLDEN: &str = include_str!("fixtures/gguf_corpus_digests.json");

fn gguf_files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(root) else {
        return;
    };
    let mut entries: Vec<_> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            gguf_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "gguf") {
            out.push(p);
        }
    }
}

type Digest = (
    String,
    BTreeMap<String, String>,
    BTreeMap<String, String>,
    BTreeMap<String, u32>,
    u64,
);

/// Everything a `Content` carries, as comparable text (the types are not all `PartialEq`).
fn digest(c: &Content) -> Digest {
    (
        format!("{:?}", c.magic),
        c.metadata
            .iter()
            .map(|(k, v)| (k.clone(), format!("{v:?}")))
            .collect(),
        c.tensor_infos
            .iter()
            .map(|(k, v)| (k.clone(), format!("{v:?}")))
            .collect(),
        c.unsupported_tensors
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect(),
        c.tensor_data_offset,
    )
}

/// FNV-1a 64 over the digest's `Debug` text: stable across Rust versions, unlike `DefaultHasher`.
fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn golden() -> BTreeMap<String, String> {
    serde_json::from_str(GOLDEN).expect("fixtures/gguf_corpus_digests.json")
}

#[test]
fn open_reproduces_the_old_readers_digest_on_every_corpus_file() {
    let Some(root) = std::env::var_os("FUEL_GGUF_CORPUS").map(PathBuf::from) else {
        skip(
            Hardware::Hub,
            Missing::device("FUEL_GGUF_CORPUS is not set (a directory of .gguf files)"),
        );
        return;
    };
    let mut files = Vec::new();
    gguf_files(&root, &mut files);
    let want = golden();
    let (mut agree, mut refused_v1, mut wrong) = (0usize, 0usize, Vec::new());
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().into_owned();
        match (Content::open(f), want.get(&name)) {
            (Ok(c), Some(w)) => {
                let got = format!("{:016x}", fnv(&format!("{:?}", digest(&c))));
                if &got == w {
                    agree += 1;
                } else {
                    wrong.push(format!("{name}: digest {got}, the old reader gave {w}"));
                }
            }
            (Ok(_), None) => wrong.push(format!("{name}: no recorded digest (new corpus file?)")),
            (Err(e), None) if e.to_string().contains("GGUF v1") => refused_v1 += 1,
            (Err(e), _) => wrong.push(format!("{name}: refused: {e}")),
        }
    }
    println!(
        "{} files: {agree} reproduce the old reader's digest, {refused_v1} v1 refused, {} wrong",
        files.len(),
        wrong.len()
    );
    for w in &wrong {
        println!("WRONG {w}");
    }
    assert!(wrong.is_empty(), "{} file(s) wrong", wrong.len());
    // Positive controls: every recorded digest was reproduced (no file silently skipped), and
    // the one v1 file is refused rather than missing.
    assert_eq!(
        agree,
        want.len(),
        "every recorded file must be present and agree"
    );
    assert_eq!(refused_v1, 1, "the corpus's single v1 file");
}

/// Sabotage sibling: the digest sees a difference when there is one.
#[test]
fn the_digest_notices_a_changed_offset() {
    let a: Digest = (
        "m".to_string(),
        BTreeMap::new(),
        BTreeMap::new(),
        BTreeMap::new(),
        32,
    );
    let mut b = a.clone();
    b.4 += 1;
    assert_ne!(fnv(&format!("{a:?}")), fnv(&format!("{b:?}")));
}
