//! # fuel-correctness-fixtures
//!
//! Distributable correctness fixtures for Fuel. A fixture is a
//! pre-validated `(op, dtype, size_class, input_seed) → expected_output`
//! tuple captured on a multi-backend system after human review of
//! any outliers. Single-backend systems use these fixtures to
//! validate their kernels without needing a peer for inline
//! pairwise consensus.
//!
//! ## Architectural role
//!
//! The Judge's pairwise consensus mechanism
//! (`fuel-core::judge` post-2026-06-07) validates kernel correctness
//! by clustering N backends' outputs and flagging outliers. It works
//! great on multi-backend systems (CPU + CUDA + Vulkan on a dev
//! box, say) but degrades to "no peer to compare against" on
//! single-backend systems.
//!
//! Fixtures fill that gap. Captured on a multi-backend system
//! (where consensus is meaningful), reviewed for outliers, and
//! shipped with Fuel as data files. Single-backend systems compare
//! their kernels' output against the captured expected output with
//! a tolerance band — same correctness signal without needing peer
//! backends locally.
//!
//! ## Architecture v0.4 reference
//!
//! See `docs/architecture/05-backend-contract.md` §Pairwise
//! consensus correctness. The "Future work" paragraph at the end
//! of that section names this crate as the optimization that lets
//! the Judge skip multi-backend re-runs on subsequent profile-cache
//! builds.
//!
//! ## What ships in v1
//!
//! - [`CorrectnessFixture`] — the in-memory representation.
//! - [`ToleranceBand`] — the rel/abs error bounds a kernel must
//!   stay within to count as matching the fixture.
//! - [`FixtureFile`] — a collection of fixtures, JSON-serializable
//!   under the `serde` feature.
//! - [`validate_against_fixture`] — compare a kernel output against
//!   a fixture's expected output + tolerance.
//!
//! ## Not yet in v1 (deferred to follow-up sessions)
//!
//! - **Capture tool**: a `fuel-capture-fixtures` binary that runs
//!   the Judge with all available backends, applies consensus
//!   clustering, surfaces outliers for human review, and writes
//!   fixture files. Substrate exists; the binary plus the
//!   review-UI workflow is its own engineering effort.
//! - **Judge integration**: load fixtures + use them in place of
//!   inline consensus when available. The Judge's existing
//!   pairwise-consensus path stays as the fallback.
//! - **Distributed fixture set**: actual fixtures captured on
//!   representative hardware (Windows + RTX 4070 dev box first,
//!   broader platforms later).
//!
//! Each of these is independently scoped; this crate ships the
//! data model + loader + validator so the consumers can plug in
//! without rediscovery.

use fuel_core_types::dispatch::{OpKind, SizeClass};
use fuel_core_types::DType;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Capture-pipeline helpers used by the `fuel-capture-fixtures`
/// binary: pairwise consensus clustering, deterministic input
/// generation, consensus-median fixture selection, and on-disk
/// emission. Public so external tooling can build alternative
/// capture front-ends without forking the data-model crate.
///
/// Gated on the `capture` feature so single-backend consumers
/// of the fixtures don't pay the `serde_json` / filesystem-helper
/// cost. Validators only need [`validate_against_fixture`]
/// + the data model.
#[cfg(feature = "capture")]
pub mod capture;

/// Tolerance band for fixture comparison. A kernel's output is
/// considered matching the fixture if every element is within
/// either the absolute or relative bound (whichever is looser).
///
/// Bounds are derived at capture time from the strictest
/// `PrecisionGuarantee::max_relative` / `max_absolute` among the
/// consensus group's kernels, with a safety multiplier (typically
/// 2×) to absorb platform rounding drift (ARM vs x86 vs Apple
/// Silicon).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ToleranceBand {
    /// Maximum relative error per output element. `1e-3` is the
    /// default for capture; tighter for ops with stricter
    /// declared `PrecisionGuarantee::max_relative`.
    pub max_relative: f64,
    /// Maximum absolute error per output element. Used as a floor
    /// for small expected values where relative-error denominators
    /// approach zero (denormals, near-zero outputs of `tanh`/`sin`/
    /// etc. at small inputs).
    pub max_absolute: f64,
}

impl ToleranceBand {
    /// The default tolerance for f32 op output: 1e-3 relative,
    /// 1e-6 absolute. Matches the `CONSENSUS_EPSILON` floor used
    /// by `fuel-core::judge::compute_pairwise_consensus`.
    pub const F32_DEFAULT: Self = Self {
        max_relative: 1e-3,
        max_absolute: 1e-6,
    };

    /// Tighter tolerance for primitives with declared
    /// `PrecisionGuarantee::max_relative` ≤ 1e-6 (e.g. IEEE-required
    /// `add`, `mul`). Captures fine-grained drift detection.
    pub const F32_STRICT: Self = Self {
        max_relative: 1e-6,
        max_absolute: 1e-9,
    };
}

/// One captured correctness datum. Identifies a specific
/// `(op, dtype, size_class, input_seed)` cell and the expected
/// output bytes after consensus review on a multi-backend
/// capture system.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CorrectnessFixture {
    pub op: OpKind,
    pub dtype: DType,
    pub size_class: SizeClass,
    /// Deterministic seed the capture tool used to generate the
    /// input tensor. Subsequent validation runs reconstruct the
    /// input from this seed, hash it, and compare to
    /// [`Self::input_hash`] as a sanity check that input generation
    /// is reproducible across hosts.
    pub input_seed: u64,
    /// BLAKE3 (or similar) hash of the regenerated input bytes.
    /// Mismatch indicates the input-generation algorithm drifted
    /// between fixture-capture and validation runs — the fixture is
    /// stale.
    pub input_hash: u64,
    /// Raw bytes of the expected output. Reinterpret as
    /// `[dtype]` elements via `bytemuck::cast_slice`. Length must
    /// equal `output_element_count * dtype.size_in_bytes()`.
    pub expected_output: Vec<u8>,
    /// Element count of the expected output. Carried explicitly so
    /// validators don't have to re-derive it from op + size_class +
    /// dtype.
    pub output_element_count: usize,
    /// Tolerance band a kernel must satisfy to count as matching.
    pub tolerance: ToleranceBand,
}

/// A collection of correctness fixtures, typically one file's worth.
///
/// Fixture files live under `fuel-correctness-fixtures/v1/` and are
/// named `{op_name}_{dtype}.json` (e.g. `matmul_f32.json`). The
/// format is JSON for human inspection; a future binary format
/// (BLAKE3-tagged, length-prefixed) can replace it if file size
/// becomes a concern.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FixtureFile {
    /// Format version. v1 is the initial shape. Bumping requires
    /// migration code in the loader.
    pub version: u32,
    /// Fixtures in this file. Typically one per size_class for a
    /// given (op, dtype); a file may carry multiple if the capture
    /// strategy bundles related cells.
    pub fixtures: Vec<CorrectnessFixture>,
}

/// Current fixture file format version. Loader migrates older
/// versions; producers always write the latest.
pub const FIXTURE_FILE_VERSION: u32 = 1;

/// Failure modes when validating a kernel output against a
/// fixture. The variants distinguish "kernel produced wrong bytes"
/// from "fixture itself looks broken / stale."
#[derive(Debug, Clone, PartialEq)]
pub enum CorrectnessDrift {
    /// Output length doesn't match the fixture's
    /// `output_element_count * dtype.size_in_bytes()`. Suggests
    /// the kernel produced wrong-shape output.
    LengthMismatch {
        expected_bytes: usize,
        actual_bytes: usize,
    },
    /// An element exceeded both `max_relative` and `max_absolute`
    /// bounds. Carries the index + offending values for debugging.
    /// `element_index` is in elements (not bytes) for the
    /// fixture's `dtype`.
    OutOfTolerance {
        element_index: usize,
        expected: f64,
        actual: f64,
        rel_err: f64,
        abs_err: f64,
    },
    /// Element-wise comparison hit a non-finite (NaN / Inf) value
    /// where the fixture expected a finite output. Distinct from
    /// `OutOfTolerance` because NaN propagation typically indicates
    /// a different bug class (uninitialized memory, divide-by-zero
    /// in an edge case).
    NonFinite {
        element_index: usize,
        actual: f64,
    },
}

impl std::fmt::Display for CorrectnessDrift {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { expected_bytes, actual_bytes } => write!(
                f,
                "length mismatch: fixture expects {expected_bytes} bytes, got {actual_bytes}",
            ),
            Self::OutOfTolerance {
                element_index,
                expected,
                actual,
                rel_err,
                abs_err,
            } => write!(
                f,
                "element {element_index} out of tolerance: \
                 expected {expected}, got {actual}, rel_err {rel_err:e}, abs_err {abs_err:e}",
            ),
            Self::NonFinite { element_index, actual } => write!(
                f,
                "element {element_index} non-finite: got {actual}",
            ),
        }
    }
}

impl std::error::Error for CorrectnessDrift {}

/// Validate a kernel output against a fixture. Returns `Ok(())` if
/// the output matches within tolerance; otherwise the first
/// drift observed (early-out — production validators that want
/// "all drifts" iterate the loop themselves).
///
/// `actual_output` is the kernel's raw bytes. The function
/// reinterprets both `actual_output` and `fixture.expected_output`
/// as `[dtype]` element arrays via `bytemuck::cast_slice` and
/// compares element-by-element.
///
/// Only f32 is wired today (matching Judge's current scope). f16 /
/// bf16 / f64 add the dtype-cast arms when those become Judge-
/// profiled.
pub fn validate_against_fixture(
    fixture: &CorrectnessFixture,
    actual_output: &[u8],
) -> Result<(), CorrectnessDrift> {
    let elem_size = fixture.dtype.size_in_bytes();
    let expected_bytes = fixture.output_element_count * elem_size;
    if actual_output.len() != expected_bytes {
        return Err(CorrectnessDrift::LengthMismatch {
            expected_bytes,
            actual_bytes: actual_output.len(),
        });
    }

    match fixture.dtype {
        DType::F32 => validate_f32(fixture, actual_output),
        _ => {
            // Future dtypes: add arms here. For now, treat
            // unknown-dtype fixtures as length-match-only; callers
            // that need element-level drift detection on f16/bf16/
            // f64 cells need a later commit to wire those casts.
            Ok(())
        }
    }
}

fn validate_f32(
    fixture: &CorrectnessFixture,
    actual_output: &[u8],
) -> Result<(), CorrectnessDrift> {
    let expected: &[f32] = bytemuck::cast_slice(&fixture.expected_output);
    let actual: &[f32] = bytemuck::cast_slice(actual_output);
    for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
        if !a.is_finite() && e.is_finite() {
            return Err(CorrectnessDrift::NonFinite {
                element_index: i,
                actual: a as f64,
            });
        }
        let e = e as f64;
        let a = a as f64;
        let abs_err = (e - a).abs();
        let denom = e.abs().max(a.abs()).max(f64::MIN_POSITIVE);
        let rel_err = abs_err / denom;
        if rel_err > fixture.tolerance.max_relative
            && abs_err > fixture.tolerance.max_absolute
        {
            return Err(CorrectnessDrift::OutOfTolerance {
                element_index: i,
                expected: e,
                actual: a,
                rel_err,
                abs_err,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_f32_fixture(expected: Vec<f32>) -> CorrectnessFixture {
        let elem_count = expected.len();
        let bytes: Vec<u8> = expected
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        CorrectnessFixture {
            op: OpKind::AddElementwise,
            dtype: DType::F32,
            size_class: SizeClass(8),
            input_seed: 42,
            input_hash: 0xdeadbeef,
            expected_output: bytes,
            output_element_count: elem_count,
            tolerance: ToleranceBand::F32_DEFAULT,
        }
    }

    fn bytes_of(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// Exact match passes validation (no drift).
    #[test]
    fn exact_match_validates_clean() {
        let fixture = make_f32_fixture(vec![1.0, 2.0, 3.0]);
        let actual = bytes_of(&[1.0, 2.0, 3.0]);
        assert!(validate_against_fixture(&fixture, &actual).is_ok());
    }

    /// Drift within the tolerance band passes.
    #[test]
    fn within_tolerance_validates_clean() {
        let fixture = make_f32_fixture(vec![1.0, 2.0, 3.0]);
        // Drift by 1e-5 relative — well under the 1e-3 default.
        let actual = bytes_of(&[1.00001, 2.00002, 3.00003]);
        assert!(validate_against_fixture(&fixture, &actual).is_ok());
    }

    /// Drift beyond tolerance returns OutOfTolerance with the
    /// first-violating index.
    #[test]
    fn out_of_tolerance_returns_first_violation() {
        let fixture = make_f32_fixture(vec![1.0, 2.0, 3.0]);
        // Drift element 1 by 1.0 (huge — well over 1e-3 relative).
        let actual = bytes_of(&[1.0, 3.0, 3.0]);
        let err = validate_against_fixture(&fixture, &actual).unwrap_err();
        match err {
            CorrectnessDrift::OutOfTolerance { element_index, .. } => {
                assert_eq!(element_index, 1);
            }
            other => panic!("expected OutOfTolerance, got {other:?}"),
        }
    }

    /// NaN in actual output → NonFinite drift, distinct from
    /// OutOfTolerance.
    #[test]
    fn nan_actual_returns_non_finite() {
        let fixture = make_f32_fixture(vec![1.0, 2.0, 3.0]);
        let actual = bytes_of(&[1.0, f32::NAN, 3.0]);
        let err = validate_against_fixture(&fixture, &actual).unwrap_err();
        match err {
            CorrectnessDrift::NonFinite { element_index, .. } => {
                assert_eq!(element_index, 1);
            }
            other => panic!("expected NonFinite, got {other:?}"),
        }
    }

    /// Wrong-length output → LengthMismatch, no element walk.
    #[test]
    fn length_mismatch_returns_length_mismatch() {
        let fixture = make_f32_fixture(vec![1.0, 2.0, 3.0]);
        let actual = bytes_of(&[1.0, 2.0]); // missing one element
        let err = validate_against_fixture(&fixture, &actual).unwrap_err();
        match err {
            CorrectnessDrift::LengthMismatch { expected_bytes, actual_bytes } => {
                assert_eq!(expected_bytes, 12);
                assert_eq!(actual_bytes, 8);
            }
            other => panic!("expected LengthMismatch, got {other:?}"),
        }
    }

    /// Absolute tolerance saves near-zero comparisons (where
    /// relative-error denominators are tiny).
    #[test]
    fn near_zero_uses_absolute_bound() {
        let mut fixture = make_f32_fixture(vec![0.0, 1e-9, 2e-9]);
        fixture.tolerance.max_absolute = 1e-6;
        // Actual drifts by 1e-7 — under the abs bound.
        let actual = bytes_of(&[1e-7, 1e-7 + 1e-9, 2e-7 + 2e-9]);
        assert!(validate_against_fixture(&fixture, &actual).is_ok());
    }

    /// Serde round-trip preserves a fixture exactly.
    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip() {
        let fixture = make_f32_fixture(vec![1.0, 2.0, 3.0]);
        let file = FixtureFile {
            version: FIXTURE_FILE_VERSION,
            fixtures: vec![fixture.clone()],
        };
        let json = serde_json::to_string(&file).expect("serialize");
        let parsed: FixtureFile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.fixtures.len(), 1);
        assert_eq!(parsed.fixtures[0], fixture);
    }
}
