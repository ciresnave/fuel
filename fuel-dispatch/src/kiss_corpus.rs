// SPDX-License-Identifier: MIT OR Apache-2.0
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
/// Ordinary (strict-inequality) minmax vectors — the max-vs-min discriminator.
const OPS_MINMAX_ORDINARY_JSON: &str =
    include_str!("../fixtures/kiss-corpus/ops-minmax-ordinary.json");
/// Signed-zero tie vectors. `+0` vs `-0` compares EQUAL, so the tie is broken by
/// operand order, not by value — a value-compare passes vacuously and only a
/// BIT comparison can see it.
const OPS_MINMAX_SIGNED_ZERO_JSON: &str =
    include_str!("../fixtures/kiss-corpus/ops-minmax-signed-zero.json");

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
    Hex {
        file: &'static str,
        tc_id: u64,
        detail: String,
    },
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
        self.vectors
            .iter()
            .filter(|v| v.op == op && v.dtype == dtype)
            .collect()
    }

    /// The set of covered `(op, dtype)` cells.
    pub fn covered_cells(&self) -> BTreeSet<(String, String)> {
        self.vectors
            .iter()
            .map(|v| (v.op.clone(), v.dtype.clone()))
            .collect()
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
    if !cleaned.len().is_multiple_of(2) {
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
    // `as_chunks::<2>()` yields `&[u8; 2]`, so the pair indexing below is
    // array indexing rather than slice indexing -- same code, no bounds risk.
    let (pairs, _odd_tail) = cleaned.as_chunks::<2>();
    for pair in pairs {
        match (hexval(pair[0]), hexval(pair[1])) {
            (Some(hi), Some(lo)) => out.push((hi << 4) | lo),
            _ => {
                return Err(CorpusError::Hex {
                    file,
                    tc_id,
                    detail: format!("non-hex digit in {s:?}"),
                });
            }
        }
    }
    Ok(out)
}

/// Parse one `kiss-oracle-vectors-v1` file into exact-byte vectors.
///
/// Extracted from `load_vendored_corpus` so every vector file goes through the
/// SAME parse, rather than a second copy that can drift. `file` is carried only
/// for error attribution — a `CorpusError` must name which file it came from
/// once more than one is loaded.
fn parse_vector_file(
    json: &'static str,
    file: &'static str,
) -> Result<Vec<CorpusVector>, CorpusError> {
    let doc: serde_json::Value =
        serde_json::from_str(json).map_err(|source| CorpusError::Json { file, source })?;
    let arr = doc
        .get("vectors")
        .and_then(|v| v.as_array())
        .ok_or_else(|| CorpusError::Schema {
            file,
            detail: "missing `vectors` array".to_string(),
        })?;

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
        let rounding = v
            .get("rounding")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let class = v
            .get("class")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let ulp_bound = v.get("ulp_bound").and_then(|x| x.as_u64()).unwrap_or(0);

        let inputs_arr =
            v.get("inputs")
                .and_then(|x| x.as_array())
                .ok_or_else(|| CorpusError::Schema {
                    file,
                    detail: format!("tcId {tc_id}: missing `inputs` array"),
                })?;
        let mut inputs = Vec::with_capacity(inputs_arr.len());
        for inp in inputs_arr {
            let bits =
                inp.get("bits")
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| CorpusError::Schema {
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
    Ok(vectors)
}

/// Load and parse the vendored corpus snapshot (manifest metadata +
/// exact-byte vectors). Pure over the embedded constants; never panics.
pub fn load_vendored_corpus() -> Result<Corpus, CorpusError> {
    // --- op_manifest.json: metadata (op names, transcendental atoms, coverage).
    let manifest: serde_json::Value =
        serde_json::from_str(OP_MANIFEST_JSON).map_err(|source| CorpusError::Json {
            file: "op_manifest.json",
            source,
        })?;
    let string_list = |key: &str| -> Vec<String> {
        manifest
            .get(key)
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    };
    let all_ops = string_list("all_ops");
    let transcendental_atoms = string_list("transcendental_atoms");
    let declared_coverage = string_list("declared_coverage_set");

    // Every vector file goes through the same parse. Order is stable so
    // `vectors` indices are reproducible across runs.
    let mut vectors = Vec::new();
    for (json, file) in [
        (OPS_ARITH_JSON, "ops-arith.json"),
        (OPS_MINMAX_ORDINARY_JSON, "ops-minmax-ordinary.json"),
        (OPS_MINMAX_SIGNED_ZERO_JSON, "ops-minmax-signed-zero.json"),
    ] {
        vectors.extend(parse_vector_file(json, file)?);
    }

    Ok(Corpus {
        all_ops,
        transcendental_atoms,
        declared_coverage,
        vectors,
    })
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
        assert!(
            corpus.covers("add", "f32"),
            "corpus should cover (add, f32)"
        );
        let cells = corpus.cells("add", "f32");
        assert_eq!(
            cells.len(),
            5,
            "expected 5 add/f32 vectors, got {}",
            cells.len()
        );
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
        assert!(
            !corpus.covers("add", "f64"),
            "corpus should NOT cover (add, f64)"
        );
        assert!(
            !corpus.covers("mul", "f32"),
            "corpus should NOT cover (mul, f32)"
        );
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

    /// The re-vendor actually loaded the minmax files — not merely dropped them
    /// in the fixture directory.
    ///
    /// This is the check that distinguishes a real re-vendor from the inert
    /// shape: `kiss_corpus.rs` reads its files through `include_str!` BY NAME,
    /// so adding JSON to the fixture directory changes nothing on its own. The
    /// directory would look current while the vectors sat unread and the gate
    /// stayed exactly as blind.
    #[test]
    fn the_minmax_vectors_are_loaded_not_merely_vendored() {
        let corpus = load_vendored_corpus().expect("vendored corpus parses");
        for op in ["max_prop", "min_prop", "fmax_ieee", "fmin_ieee"] {
            let n = corpus.vectors.iter().filter(|v| v.op == op).count();
            assert_eq!(
                n, 18,
                "expected 18 vectors for `{op}` (6 ordinary + 12 signed-zero); \
                 got {n}. A zero here means the file is in the directory but not \
                 in `include_str!`."
            );
        }
        // Control: the pre-existing arith vectors are still present and did not
        // move, so the count above is an ADDITION rather than a replacement.
        assert_eq!(
            corpus.cells("add", "f32").len(),
            5,
            "the original 5 add/f32 vectors must survive the re-vendor"
        );
    }

    /// ⚠️ TRIPWIRE, AND IT IS SUPPOSED TO PASS TODAY AND FAIL LATER.
    ///
    /// The vendored corpus CANNOT distinguish NaN-propagating minmax
    /// (`max_prop`/`min_prop`) from IEEE minmax (`fmax_ieee`/`fmin_ieee`).
    /// Measured at KISS `f4952b4c`: of the 18 distinct `(dtype, input-pair)`
    /// cells, ALL 18 carry all four ops and ZERO of them give the prop and ieee
    /// forms different expected bits.
    ///
    /// That is structural, not accidental — the corpus's own provenance note
    /// says so: §6.6 signed-zero equality makes both comparisons true on a ±0
    /// tie, so operand `a` wins every tie in all four ops. NaN is the axis on
    /// which the two families differ, and the corpus contains NO NaN INPUT AT
    /// ALL (77 vectors, 0 with a NaN operand).
    ///
    /// So these vectors buy real coverage — signed-zero tie-breaking and
    /// max-vs-min — and they do NOT close the defect GAP-236 is about: an
    /// `fmaxf` mis-lifted as a NaN-propagating `Max` passes every one of them.
    ///
    /// WHY THIS IS A TEST AND NOT A COMMENT: a comment recording "the corpus is
    /// blind here" is read only by someone already standing here, and nothing
    /// fires when it stops being true. When KISS ships the NaN vectors, this
    /// test GOES RED — which is the signal to re-measure and close the row.
    /// A prose note would have gone silently stale instead.
    #[test]
    fn the_corpus_cannot_yet_discriminate_prop_from_ieee_minmax() {
        use std::collections::BTreeMap;
        let corpus = load_vendored_corpus().expect("vendored corpus parses");

        // Group by (dtype, input bytes) so the four ops line up per cell.
        let mut cells: BTreeMap<(String, Vec<Vec<u8>>), BTreeMap<String, Vec<u8>>> =
            BTreeMap::new();
        for v in &corpus.vectors {
            if matches!(
                v.op.as_str(),
                "max_prop" | "min_prop" | "fmax_ieee" | "fmin_ieee"
            ) {
                cells
                    .entry((v.dtype.clone(), v.inputs.clone()))
                    .or_default()
                    .insert(v.op.clone(), v.expected.clone());
            }
        }

        let discriminating = cells
            .values()
            .filter(|ops| {
                let max_differs = matches!(
                    (ops.get("max_prop"), ops.get("fmax_ieee")),
                    (Some(a), Some(b)) if a != b
                );
                let min_differs = matches!(
                    (ops.get("min_prop"), ops.get("fmin_ieee")),
                    (Some(a), Some(b)) if a != b
                );
                max_differs || min_differs
            })
            .count();

        // POSITIVE CONTROL: the corpus is not simply inert. It DOES separate max
        // from min, so a zero above is a statement about the prop/ieee axis
        // specifically, not about the grouping being broken.
        let max_vs_min = cells
            .values()
            .filter(|ops| {
                matches!(
                    (ops.get("max_prop"), ops.get("min_prop")),
                    (Some(a), Some(b)) if a != b
                )
            })
            .count();
        assert!(
            max_vs_min > 0,
            "control failed: the corpus should separate max from min on at least \
             one cell — if this is 0 the cell grouping is broken and the \
             prop-vs-ieee count below means nothing"
        );

        assert_eq!(
            discriminating,
            0,
            "the corpus now HAS a cell where prop and ieee minmax differ ({discriminating} \
             of {} cells). This test existing and passing recorded a KNOWN BLIND SPOT; \
             a red here is GOOD NEWS — NaN vectors have landed. Re-measure the \
             discrimination, wire the assertion the other way, and close the \
             registry row that tracks this.",
            cells.len()
        );
    }
}
