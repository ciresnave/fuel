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
/// NaN-operand vectors for the sign-edit moves (`neg`/`abs`/`copysign`) on the narrow
/// float dtypes (KISS-OPS-6.16-0009). A moved NaN keeps its payload; only the sign bit
/// is edited, so a kernel that canonicalises a NaN fails here and nowhere else.
const OPS_NARROW_MOVE_NAN_JSON: &str =
    include_str!("../fixtures/kiss-corpus/ops-narrow-move-nan.json");
/// NaN-operand vectors for `max_prop`/`min_prop`/`fmax_ieee`/`fmin_ieee` on the narrow
/// float dtypes: the cells where the propagating and IEEE families DIFFER, which the
/// signed-zero file structurally cannot supply (see the discriminator test below).
const OPS_NARROW_SELECT_NAN_JSON: &str =
    include_str!("../fixtures/kiss-corpus/ops-narrow-select-nan.json");
/// NaN-operand vectors for the transcendental atoms (`exp`/`log`/`sin`), class `ULP`
/// with a non-zero bound: a computed NaN is canonical, not payload-preserving.
const OPS_TRANSCENDENTAL_NAN_JSON: &str =
    include_str!("../fixtures/kiss-corpus/ops-transcendental-nan.json");
/// NaN-operand vectors for the four minmax ops on `f32` and `f64` -- the cells the
/// narrow-dtype file could not supply. This is what closes the f32/f64 half of the
/// prop-vs-ieee blind spot (GAP-261).
const OPS_MINMAX_NAN_JSON: &str = include_str!("../fixtures/kiss-corpus/ops-minmax-nan.json");
/// `floor`/`ceil`/`trunc`/`round_even` on `f32`: the vectors `trunc` never had
/// (GAP-300 recorded that Fuel's `trunc` expansion was caught by no KISS vector).
const OPS_ROUNDING_JSON: &str = include_str!("../fixtures/kiss-corpus/ops-rounding.json");

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
    ///
    /// **This is the predicate an activation check wants** — it asks whether the
    /// corpus can actually verify this cell. Contrast [`Corpus::declares_op`],
    /// which answers a different question despite the closer-sounding name.
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
    ///
    /// ⚠️ **NOT the predicate to gate a conformance or activation check on, and
    /// its name is the trap.** `declared_coverage_set` is KISS's **scope marker
    /// for the current minting slice** — their pipeline plan defines it as *"op
    /// names in scope for the CURRENT slice"* and says *"it grows as slices
    /// land"*. It is NOT an inventory of which ops have vectors.
    ///
    /// Measured against the vendored corpus (KISS `f4952b4c`), the two answers
    /// differ for 12 of 13 cells:
    ///
    /// ```text
    ///              declares_op   covers(f32)
    /// add             true          true
    /// max_prop        FALSE         true      (+ bf16, f64)
    /// min_prop        FALSE         true      (+ bf16, f64)
    /// fmax_ieee       FALSE         true      (+ bf16, f64)
    /// fmin_ieee       FALSE         true      (+ bf16, f64)
    /// ```
    ///
    /// **Both answers are correct; they answer different questions.** KISS's own
    /// §6.5-0008 test asserts `declared ⊆ covered` — one-directional by design,
    /// so vectors beyond the declared slice are expected, not an inconsistency.
    ///
    /// **So gating activation on this would silently verify one op out of five
    /// and look like it worked.** Use [`Corpus::covers`] for "can the corpus
    /// check this cell". This one is for asking what slice upstream has minted.
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
        (OPS_NARROW_MOVE_NAN_JSON, "ops-narrow-move-nan.json"),
        (OPS_NARROW_SELECT_NAN_JSON, "ops-narrow-select-nan.json"),
        (OPS_TRANSCENDENTAL_NAN_JSON, "ops-transcendental-nan.json"),
        (OPS_MINMAX_NAN_JSON, "ops-minmax-nan.json"),
        (OPS_ROUNDING_JSON, "ops-rounding.json"),
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
        // 18 per op from the ordinary (6) + signed-zero (12) files, plus the narrow
        // NaN select file (8 per prop op, 12 per ieee op) and `ops-minmax-nan.json`
        // (f32+f64: 12 per prop op, 10 per ieee op). Counted from the vendored JSON
        // at KISS `0af5d3b6`, not predicted.
        for (op, want) in [
            ("max_prop", 38usize),
            ("min_prop", 38),
            ("fmax_ieee", 40),
            ("fmin_ieee", 40),
        ] {
            let n = corpus.vectors.iter().filter(|v| v.op == op).count();
            assert_eq!(
                n, want,
                "expected {want} vectors for `{op}`; got {n}. A shortfall of 8 \
                 (prop) or 12 (ieee) means `ops-narrow-select-nan.json` is in the \
                 directory but not in `include_str!`."
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

    /// `(dtype, input-bytes) -> {op -> expected-bytes}`: one cell per distinct
    /// input tuple, holding each minmax op's expected output for it.
    type MinmaxCells = std::collections::BTreeMap<
        (String, Vec<Vec<u8>>),
        std::collections::BTreeMap<String, Vec<u8>>,
    >;

    /// Group minmax vectors into `(dtype, input-bytes) -> {op -> expected}`
    /// cells, so the four ops line up per cell and can be compared.
    fn minmax_cells(corpus: &Corpus) -> MinmaxCells {
        let mut cells = MinmaxCells::new();
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
        cells
    }

    /// True iff this cell gives the prop and ieee forms different expected bits.
    fn cell_discriminates_prop_from_ieee(
        ops: &std::collections::BTreeMap<String, Vec<u8>>,
    ) -> bool {
        let differs =
            |a: &str, b: &str| matches!((ops.get(a), ops.get(b)), (Some(x), Some(y)) if x != y);
        differs("max_prop", "fmax_ieee") || differs("min_prop", "fmin_ieee")
    }

    /// True iff `bytes` (big-endian value bytes) is a NaN in `dtype`. Only the
    /// dtypes the minmax corpus uses are decoded; anything else is `false`.
    fn is_nan_bits(dtype: &str, bytes: &[u8]) -> bool {
        match (dtype, bytes) {
            ("bf16", [h, l]) => {
                let v = u16::from_be_bytes([*h, *l]);
                v & 0x7F80 == 0x7F80 && v & 0x007F != 0
            }
            ("f16", [h, l]) => {
                let v = u16::from_be_bytes([*h, *l]);
                v & 0x7C00 == 0x7C00 && v & 0x03FF != 0
            }
            ("f8e5m2", [b]) => b & 0x7C == 0x7C && b & 0x03 != 0,
            // E4M3FN has no infinities: S.1111.111 is the only NaN encoding.
            ("f8e4m3fn", [b]) => b & 0x7F == 0x7F,
            ("f32", [a, b, c, d]) => {
                let v = u32::from_be_bytes([*a, *b, *c, *d]);
                v & 0x7F80_0000 == 0x7F80_0000 && v & 0x007F_FFFF != 0
            }
            ("f64", b) if b.len() == 8 => {
                let v = u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
                v & 0x7FF0_0000_0000_0000 == 0x7FF0_0000_0000_0000 && v & 0x000F_FFFF_FFFF_FFFF != 0
            }
            _ => false,
        }
    }

    /// The corpus now SEPARATES NaN-propagating minmax (`max_prop`/`min_prop`)
    /// from IEEE minmax (`fmax_ieee`/`fmin_ieee`) -- on the narrow dtypes only.
    ///
    /// This test replaced `the_corpus_cannot_yet_discriminate_prop_from_ieee_minmax`,
    /// a tripwire that was written to go red the day KISS shipped NaN vectors. It
    /// went red when `ops-narrow-select-nan.json` was vendored (KISS `5b28bf51`):
    /// 8 of the 30 distinct `(dtype, input-pair)` cells now give the two families
    /// different expected bits. Every one is a `(sNaN, finite)` or `(finite, sNaN)`
    /// pair -- the NaN axis, which the signed-zero file structurally cannot reach
    /// (§6.6 signed-zero equality makes `a` win every tie in all four ops).
    #[test]
    fn the_corpus_discriminates_prop_from_ieee_minmax_on_the_nan_axis() {
        let corpus = load_vendored_corpus().expect("vendored corpus parses");
        let cells = minmax_cells(&corpus);

        // CONTROL: cell grouping works -- the corpus still separates max from min.
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
            "control failed: no cell separates max from min, so the cell grouping is broken"
        );

        let discriminating: Vec<_> = cells
            .iter()
            .filter(|(_, ops)| cell_discriminates_prop_from_ieee(ops))
            .collect();
        assert_eq!(
            discriminating.len(),
            18,
            "expected 18 prop-vs-ieee discriminating cells: 2 orderings x 4 narrow dtypes \
             (8, ops-narrow-select-nan) + 5 f32 + 5 f64 (ops-minmax-nan)"
        );
        for ((dtype, inputs), _) in &discriminating {
            assert!(
                inputs.iter().any(|b| is_nan_bits(dtype, b)),
                "a discriminating cell with no NaN operand ({dtype} {inputs:?}) would mean the \
                 separation is NOT on the NaN axis, so the reading above is wrong"
            );
        }
        let mut dtypes: Vec<&str> = discriminating
            .iter()
            .map(|((d, _), _)| d.as_str())
            .collect();
        dtypes.sort_unstable();
        dtypes.dedup();
        assert_eq!(dtypes, ["bf16", "f16", "f32", "f64", "f8e4m3fn", "f8e5m2"]);
    }

    /// The f32/f64 half of the blind spot is CLOSED by `ops-minmax-nan.json`.
    ///
    /// This replaced `the_corpus_still_cannot_discriminate_prop_from_ieee_minmax_for_f32_and_f64`,
    /// a tripwire written to go red when KISS shipped f32/f64 NaN select vectors.
    /// It went red on vendoring `ops-minmax-nan.json` (KISS `0af5d3b6`): 5 f32 and
    /// 5 f64 cells now give the propagating and IEEE families different bits, so an
    /// `fmaxf`-class mis-lift as a propagating `Max` is no longer invisible on the
    /// two dtypes that matter most.
    #[test]
    fn the_corpus_discriminates_prop_from_ieee_minmax_for_f32_and_f64() {
        let corpus = load_vendored_corpus().expect("vendored corpus parses");
        let cells = minmax_cells(&corpus);

        // CONTROL: f32 and f64 cells exist and separate max from min, so a count
        // below is about the prop/ieee axis, not about missing dtypes.
        for dtype in ["f32", "f64"] {
            let separates_max_min = cells.iter().any(|((d, _), ops)| {
                d == dtype
                    && matches!(
                        (ops.get("max_prop"), ops.get("min_prop")),
                        (Some(a), Some(b)) if a != b
                    )
            });
            assert!(separates_max_min, "control failed for {dtype}");
        }

        for dtype in ["f32", "f64"] {
            let hits: Vec<_> = cells
                .iter()
                .filter(|((d, _), ops)| d == dtype && cell_discriminates_prop_from_ieee(ops))
                .collect();
            assert_eq!(
                hits.len(),
                5,
                "expected 5 {dtype} prop-vs-ieee discriminating cells, got {}",
                hits.len()
            );
            for ((d, inputs), _) in &hits {
                assert!(
                    inputs.iter().any(|b| is_nan_bits(d, b)),
                    "an f32/f64 discriminating cell with no NaN operand: {d} {inputs:?}"
                );
            }
        }
    }

    /// The three NaN files were loaded, not merely vendored: every cell they
    /// introduce is reachable through `covers`, and the total is the sum of all
    /// eight vector files (5 + 24 + 48 + 30 + 40 + 13 + 44 + 72).
    #[test]
    fn the_nan_vectors_are_loaded_not_merely_vendored() {
        let corpus = load_vendored_corpus().expect("vendored corpus parses");
        assert_eq!(corpus.vectors.len(), 276);
        for (op, dtype) in [
            ("neg", "bf16"),
            ("abs", "f8e4m3fn"),
            ("copysign", "f16"),
            ("exp", "f32"),
            ("log", "f64"),
            ("sin", "f32"),
        ] {
            assert!(
                corpus.covers(op, dtype),
                "({op}, {dtype}) is in a NaN file but not covered: the file is not in `include_str!`"
            );
        }
        // The transcendental file is the only `ULP`-class one: a computed NaN is
        // canonical, so these cells carry a non-zero bound, unlike every
        // exact-byte cell.
        let ulp: Vec<_> = corpus.vectors.iter().filter(|v| v.class == "ULP").collect();
        assert_eq!(ulp.len(), 13);
        assert!(ulp.iter().all(|v| v.ulp_bound > 0));
    }
}
