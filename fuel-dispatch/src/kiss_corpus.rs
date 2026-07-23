//! Reader for the vendored KISS conformance corpus (`kiss-oracle-vectors-v1`).
//!
//! This is the DATA-READER half of the kiss-ref verdict seam. It parses the
//! pinned snapshot under `fuel-dispatch/fixtures/kiss-corpus/` (KISS `main`
//! @ `c9153b2`; see that dir's `PROVENANCE.md`) into an in-memory [`Corpus`] of
//! per-`(op, dtype, input-vector)` **exact-byte** reference cells.
//!
//! ## Deliberately NOT wired into [`crate::jit_ingest::corpus_verdict`]
//!
//! The corpus is an **oracle** — fixed input bit patterns and their single
//! correct output bit pattern — not an `(op, dtype) → adopt/reject` table.
//! `corpus_verdict`'s seam is `(op, dtype, seed) -> Option<CorpusOutcome>`: it
//! receives NO candidate output, and its `seed` selects a *random probe*
//! (`jit_ingest_probe::probe_from_operands`) that is disjoint from these fixed
//! corpus inputs. Turning this oracle into a candidate verdict requires
//! re-running the candidate on the corpus's own input vectors and comparing
//! byte-exact — a seam change out of scope for the reader increment (A4b). So
//! `corpus_verdict` stays dormant (`None`) and this reader is staged for the
//! corrected seam. Rationale + the required seam shape are recorded in
//! `docs/design-notes/2026-07-23-kiss-corpus-verdict-seam-mismatch.md`.
//!
//! Byte convention: `bits` in the corpus are the value's bytes **most-significant
//! first** (big-endian value bytes). They are stored here verbatim, as parsed.
//! A consumer comparing against Fuel's little-endian tensor storage must swap.

// Staged reader: its only intended runtime consumer (`corpus_verdict`) is
// dormant pending the seam correction above, so the public surface is currently
// exercised only by the in-module tests. Suppress dead-code noise in the
// non-test `--features jit` build until the corrected seam consumes it.
#![allow(dead_code)]

use std::collections::BTreeSet;
use thiserror::Error;

/// The pinned corpus snapshot, embedded at compile time (no runtime cwd
/// dependence). Provenance: `fuel-dispatch/fixtures/kiss-corpus/PROVENANCE.md`.
const OP_MANIFEST_JSON: &str = include_str!("../fixtures/kiss-corpus/op_manifest.json");
const OPS_ARITH_JSON: &str = include_str!("../fixtures/kiss-corpus/ops-arith.json");

/// A never-panic parse/schema failure. The embedded snapshot is a build-time
/// constant, so in practice `load_vendored_corpus` always succeeds — the
/// `Result` keeps the reader honest (a corrupt re-vendoring surfaces as a typed
/// error, never a crash).
#[derive(Debug, Error)]
pub enum CorpusError {
    #[error("corpus JSON parse error in {file}: {source}")]
    Json {
        file: &'static str,
        #[source]
        source: serde_json::Error,
    },
    #[error("corpus schema error in {file}: {detail}")]
    Schema { file: &'static str, detail: String },
    #[error("corpus hex parse error in {file} (tcId {tc_id}): {detail}")]
    Hex { file: &'static str, tc_id: u64, detail: String },
}

/// One `kiss-oracle-vectors-v1` test vector: a fixed input tuple and its single
/// correct output, all as big-endian value bytes.
#[derive(Debug, Clone)]
pub struct CorpusVector {
    pub tc_id: u64,
    pub op: String,
    pub dtype: String,
    pub rounding: String,
    /// Input operand bytes, in the corpus's declared role order (big-endian).
    pub inputs: Vec<Vec<u8>>,
    /// Expected output bytes (big-endian).
    pub expected: Vec<u8>,
    /// Vector class, e.g. `"exact-byte"`.
    pub class: String,
    /// The vector's ULP bound (`0` for exact-byte cells).
    pub ulp_bound: u64,
}

/// The parsed corpus: manifest metadata + the flattened vector list.
#[derive(Debug, Clone, Default)]
pub struct Corpus {
    /// Every op named by the spec (`op_manifest.all_ops`).
    pub all_ops: Vec<String>,
    /// The transcendental atom set (`op_manifest.transcendental_atoms`).
    pub transcendental_atoms: Vec<String>,
    /// Ops the corpus DECLARES it covers (`op_manifest.declared_coverage_set`).
    pub declared_coverage: Vec<String>,
    /// All exact-byte vectors (across every covered `(op, dtype)`).
    pub vectors: Vec<CorpusVector>,
}

impl Corpus {
    /// True iff at least one vector exists for this `(op, dtype)` cell.
    pub fn covers(&self, op: &str, dtype: &str) -> bool {
        self.vectors.iter().any(|v| v.op == op && v.dtype == dtype)
    }

    /// Every vector for this `(op, dtype)` cell (empty when uncovered).
    pub fn cells(&self, op: &str, dtype: &str) -> Vec<&CorpusVector> {
        self.vectors.iter().filter(|v| v.op == op && v.dtype == dtype).collect()
    }

    /// The set of covered `(op, dtype)` cells.
    pub fn covered_cells(&self) -> BTreeSet<(String, String)> {
        self.vectors.iter().map(|v| (v.op.clone(), v.dtype.clone())).collect()
    }

    /// True iff the manifest's `declared_coverage_set` names this op.
    pub fn declares_op(&self, op: &str) -> bool {
        self.declared_coverage.iter().any(|o| o == op)
    }
}

/// Parse `"3F 80 00 00"` (spaces and `·` are grouping marks per the corpus
/// header) into raw bytes. Never panics — an odd length or non-hex digit is a
/// typed [`CorpusError::Hex`].
fn parse_hex_bytes(s: &str, file: &'static str, tc_id: u64) -> Result<Vec<u8>, CorpusError> {
    let cleaned: Vec<u8> = s
        .bytes()
        .filter(|b| !b.is_ascii_whitespace() && *b != b'.')
        // '·' (U+00B7) is multibyte in UTF-8; drop its bytes too.
        .filter(|b| *b != 0xC2 && *b != 0xB7)
        .collect();
    if cleaned.len() % 2 != 0 {
        return Err(CorpusError::Hex {
            file,
            tc_id,
            detail: format!("odd hex-digit count {} in {s:?}", cleaned.len()),
        });
    }
    let hexval = |b: u8| -> Option<u8> {
        match b {
            b'0'..=b'9' => Some(b - b'0'),
            b'a'..=b'f' => Some(b - b'a' + 10),
            b'A'..=b'F' => Some(b - b'A' + 10),
            _ => None,
        }
    };
    let mut out = Vec::with_capacity(cleaned.len() / 2);
    for pair in cleaned.chunks_exact(2) {
        match (hexval(pair[0]), hexval(pair[1])) {
            (Some(hi), Some(lo)) => out.push((hi << 4) | lo),
            _ => {
                return Err(CorpusError::Hex {
                    file,
                    tc_id,
                    detail: format!("non-hex digit in {s:?}"),
                })
            }
        }
    }
    Ok(out)
}

/// Load and parse the vendored corpus snapshot (manifest metadata +
/// exact-byte vectors). Pure over the embedded constants; never panics.
pub fn load_vendored_corpus() -> Result<Corpus, CorpusError> {
    // --- op_manifest.json: metadata (op names, transcendental atoms, coverage).
    let manifest: serde_json::Value = serde_json::from_str(OP_MANIFEST_JSON)
        .map_err(|source| CorpusError::Json { file: "op_manifest.json", source })?;
    let string_list = |key: &str| -> Vec<String> {
        manifest
            .get(key)
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default()
    };
    let all_ops = string_list("all_ops");
    let transcendental_atoms = string_list("transcendental_atoms");
    let declared_coverage = string_list("declared_coverage_set");

    // --- ops-arith.json: the exact-byte vectors.
    let arith: serde_json::Value = serde_json::from_str(OPS_ARITH_JSON)
        .map_err(|source| CorpusError::Json { file: "ops-arith.json", source })?;
    let arr = arith
        .get("vectors")
        .and_then(|v| v.as_array())
        .ok_or_else(|| CorpusError::Schema {
            file: "ops-arith.json",
            detail: "missing `vectors` array".to_string(),
        })?;

    let file = "ops-arith.json";
    let mut vectors = Vec::with_capacity(arr.len());
    for v in arr {
        let tc_id = v.get("tcId").and_then(|x| x.as_u64()).unwrap_or(0);
        let req_str = |key: &str| -> Result<String, CorpusError> {
            v.get(key)
                .and_then(|x| x.as_str())
                .map(String::from)
                .ok_or_else(|| CorpusError::Schema {
                    file,
                    detail: format!("tcId {tc_id}: missing string field `{key}`"),
                })
        };
        let op = req_str("op")?;
        let dtype = req_str("dtype")?;
        let rounding = v.get("rounding").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let class = v.get("class").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let ulp_bound = v.get("ulp_bound").and_then(|x| x.as_u64()).unwrap_or(0);

        let inputs_arr =
            v.get("inputs").and_then(|x| x.as_array()).ok_or_else(|| CorpusError::Schema {
                file,
                detail: format!("tcId {tc_id}: missing `inputs` array"),
            })?;
        let mut inputs = Vec::with_capacity(inputs_arr.len());
        for inp in inputs_arr {
            let bits =
                inp.get("bits").and_then(|x| x.as_str()).ok_or_else(|| CorpusError::Schema {
                    file,
                    detail: format!("tcId {tc_id}: input missing `bits`"),
                })?;
            inputs.push(parse_hex_bytes(bits, file, tc_id)?);
        }

        let expected_bits = v
            .get("expected")
            .and_then(|x| x.get("bits"))
            .and_then(|x| x.as_str())
            .ok_or_else(|| CorpusError::Schema {
                file,
                detail: format!("tcId {tc_id}: missing `expected.bits`"),
            })?;
        let expected = parse_hex_bytes(expected_bits, file, tc_id)?;

        vectors.push(CorpusVector {
            tc_id,
            op,
            dtype,
            rounding,
            inputs,
            expected,
            class,
            ulp_bound,
        });
    }

    Ok(Corpus { all_ops, transcendental_atoms, declared_coverage, vectors })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendored_corpus_covers_add_f32_with_five_exact_byte_vectors() {
        let corpus = load_vendored_corpus().expect("vendored corpus parses");
        // op_manifest declares `add` as the covered op.
        assert!(
            corpus.declares_op("add"),
            "op_manifest declared_coverage_set should include add"
        );
        // ops-arith.json has exactly 5 add/f32 exact-byte vectors.
        assert!(corpus.covers("add", "f32"), "corpus should cover (add, f32)");
        let cells = corpus.cells("add", "f32");
        assert_eq!(cells.len(), 5, "expected 5 add/f32 vectors, got {}", cells.len());
        for c in &cells {
            assert_eq!(c.class, "exact-byte");
            assert_eq!(c.ulp_bound, 0);
        }
        // tcId 4: 1.0 + 1.0 = 2.0, MSB-first (big-endian) value bytes.
        let tc4 = cells.iter().find(|c| c.tc_id == 4).expect("tcId 4 present");
        assert_eq!(
            tc4.inputs,
            vec![vec![0x3F, 0x80, 0x00, 0x00], vec![0x3F, 0x80, 0x00, 0x00]]
        );
        assert_eq!(tc4.expected, vec![0x40, 0x00, 0x00, 0x00]);
        // Uncovered cells report no coverage (None-for-everything-else contract).
        assert!(!corpus.covers("add", "f64"), "corpus should NOT cover (add, f64)");
        assert!(!corpus.covers("mul", "f32"), "corpus should NOT cover (mul, f32)");
        assert!(corpus.cells("mul", "f32").is_empty());
    }

    #[test]
    fn corpus_reader_exposes_manifest_metadata() {
        let corpus = load_vendored_corpus().expect("vendored corpus parses");
        assert!(corpus.all_ops.contains(&"add".to_string()));
        assert!(corpus.transcendental_atoms.contains(&"exp".to_string()));
        // `add` is exact-class, so it is NOT a transcendental atom.
        assert!(!corpus.transcendental_atoms.contains(&"add".to_string()));
    }

    #[test]
    fn hex_parser_rejects_bad_input_without_panicking() {
        assert_eq!(
            parse_hex_bytes("3F 80 00 00", "t", 0).unwrap(),
            vec![0x3F, 0x80, 0x00, 0x00]
        );
        assert!(parse_hex_bytes("3F 8", "t", 0).is_err()); // odd digit count
        assert!(parse_hex_bytes("ZZ", "t", 0).is_err()); // non-hex
    }
}
