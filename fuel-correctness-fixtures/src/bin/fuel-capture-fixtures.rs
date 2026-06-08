//! `fuel-capture-fixtures` — produce distributable correctness
//! fixtures by running the Judge's op×dtype×size matrix across
//! every available backend, applying pairwise consensus clustering,
//! and writing per-(op, dtype) fixture files.
//!
//! ## Architectural role
//!
//! Single-backend systems can't run inline pairwise consensus
//! (no peer to disagree with). The capture binary runs on a
//! multi-backend rig (CPU + CUDA + Vulkan on a dev box) and emits
//! pre-validated fixtures that single-backend systems can validate
//! against later. See `fuel-correctness-fixtures` crate docs for
//! the broader pipeline.
//!
//! ## Modes
//!
//! `--list-cells`
//!   Print the capture cells that would be measured, without
//!   doing any I/O.
//!
//! `--mock --out-dir <DIR>`
//!   Synthesize fake per-backend measurements (CPU + CUDA-mock +
//!   Vulkan-mock all agreeing on a deterministic stub output) and
//!   run them end-to-end through the consensus + emission
//!   pipeline. Writes fixture files under `<DIR>/v1/...`. The
//!   `--mock` mode is the smoke test the binary's CI runs in
//!   place of live hardware — verifies the producer→file path
//!   stays wired across refactors.
//!
//! Live-hardware capture (the actual `--live` mode) is the next
//! gate: it requires wiring against `fuel-core::judge::Judge`'s
//! `run()` API, which pulls in the entire backend stack. That's a
//! separate engineering session — the data model + helpers + this
//! binary are the substrate.
//!
//! ## Exit codes
//!
//! - `0`: success (fixtures written, or `--list-cells` printed).
//! - `1`: invalid arguments / I/O error.
//! - `2`: capture ran but one or more cells failed consensus
//!   (review report written; non-fatal).

use std::path::PathBuf;
use std::process::ExitCode;

use fuel_correctness_fixtures::capture::{
    default_tolerance_for, fixture_from_consensus, group_fixtures_for_emission,
    representative_capture_matrix, write_fixture_file, CaptureCell, ConsensusDecision,
    MeasuredOutput, ReviewEntry, ReviewReport,
};
use fuel_correctness_fixtures::CorrectnessFixture;

const HELP_TEXT: &str = "\
fuel-capture-fixtures — produce distributable correctness fixtures

USAGE:
    fuel-capture-fixtures [OPTIONS]

OPTIONS:
    -h, --help              Print this help and exit
    --list-cells            Print the capture matrix and exit
    --mock                  Run with mock (synthesized) backend outputs
                            for smoke-testing the pipeline end-to-end
    --out-dir <PATH>        Where to write fixture files (default: ./fixtures)
    --review-file <PATH>    Where to write the human-review report when
                            consensus fails (default: <out-dir>/v1/REVIEW.md)

EXAMPLES:
    fuel-capture-fixtures --list-cells
    fuel-capture-fixtures --mock --out-dir target/fixtures-smoke
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse_args(&args) {
        Ok(Cli::Help) => {
            println!("{HELP_TEXT}");
            ExitCode::SUCCESS
        }
        Ok(Cli::ListCells) => {
            print_capture_cells();
            ExitCode::SUCCESS
        }
        Ok(Cli::Mock { out_dir, review_file }) => match run_mock(&out_dir, review_file.as_deref()) {
            Ok(RunSummary { fixtures_written, cells_skipped }) => {
                eprintln!(
                    "fuel-capture-fixtures: mock run wrote {fixtures_written} fixture(s) to {}",
                    out_dir.display(),
                );
                if cells_skipped > 0 {
                    eprintln!(
                        "fuel-capture-fixtures: {cells_skipped} cell(s) skipped pending human review",
                    );
                    ExitCode::from(2)
                } else {
                    ExitCode::SUCCESS
                }
            }
            Err(e) => {
                eprintln!("fuel-capture-fixtures: {e}");
                ExitCode::from(1)
            }
        },
        Err(e) => {
            eprintln!("fuel-capture-fixtures: {e}");
            eprintln!();
            eprintln!("{HELP_TEXT}");
            ExitCode::from(1)
        }
    }
}

#[derive(Debug)]
enum Cli {
    Help,
    ListCells,
    Mock {
        out_dir: PathBuf,
        review_file: Option<PathBuf>,
    },
}

fn parse_args(args: &[String]) -> Result<Cli, String> {
    if args.is_empty() {
        return Ok(Cli::Help);
    }
    let mut iter = args.iter();
    let mut mock = false;
    let mut list = false;
    let mut out_dir: Option<PathBuf> = None;
    let mut review_file: Option<PathBuf> = None;
    while let Some(a) = iter.next() {
        match a.as_str() {
            "-h" | "--help" => return Ok(Cli::Help),
            "--list-cells" => list = true,
            "--mock" => mock = true,
            "--out-dir" => {
                let v = iter.next().ok_or_else(|| "--out-dir requires a value".to_string())?;
                out_dir = Some(PathBuf::from(v));
            }
            "--review-file" => {
                let v = iter.next().ok_or_else(|| "--review-file requires a value".to_string())?;
                review_file = Some(PathBuf::from(v));
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if list {
        return Ok(Cli::ListCells);
    }
    if mock {
        return Ok(Cli::Mock {
            out_dir: out_dir.unwrap_or_else(|| PathBuf::from("fixtures")),
            review_file,
        });
    }
    // No mode specified — default to help. Live mode is reserved
    // for a future session that wires the Judge dependency.
    Ok(Cli::Help)
}

fn print_capture_cells() {
    let cells = representative_capture_matrix();
    println!("# Capture matrix ({} cells)", cells.len());
    println!("# (op, dtype, size_class, input_seed)");
    for cell in cells {
        println!(
            "{:<24} {:<8} sc={:<3} seed={}",
            cell.op.as_str(),
            format!("{:?}", cell.dtype),
            cell.size_class.0,
            cell.input_seed,
        );
    }
}

/// Synthesize three "backends" all agreeing on a deterministic
/// stub output. This is what a clean capture looks like end-to-end
/// — every cell produces consensus and a fixture is emitted.
/// Useful as a CI smoke test for the producer pipeline (without
/// the cost of running real backends).
fn mock_outputs_for(cell: CaptureCell) -> Vec<MeasuredOutput> {
    // Stub output: a small synthetic vector seeded off the cell.
    // The size mirrors the cell's logical output size in
    // miniature — three elements is enough for consensus to be
    // meaningful while keeping mock-mode fast.
    let base = (cell.input_seed as f32 * 1e-6).sin();
    let stub: Vec<f32> = (0..3).map(|i| base + i as f32 * 0.01).collect();
    vec![
        MeasuredOutput {
            backend_label: "cpu:portable".to_string(),
            kernel_source: "portable-cpu".to_string(),
            output: stub.clone(),
        },
        MeasuredOutput {
            backend_label: "cuda:0".to_string(),
            kernel_source: "cublas".to_string(),
            output: stub.clone(),
        },
        MeasuredOutput {
            backend_label: "vulkan:0".to_string(),
            kernel_source: "slang".to_string(),
            output: stub,
        },
    ]
}

struct RunSummary {
    fixtures_written: usize,
    cells_skipped: usize,
}

fn run_mock(out_dir: &std::path::Path, review_file: Option<&std::path::Path>) -> std::io::Result<RunSummary> {
    let cells = representative_capture_matrix();
    let mut fixtures: Vec<CorrectnessFixture> = Vec::new();
    let mut review = ReviewReport::new();
    for cell in cells {
        let outputs = mock_outputs_for(cell);
        // Mock-mode "input" is a tiny stub; live mode would call
        // `deterministic_f32_input(cell.input_seed,
        // size_class_to_elem_count(cell.size_class))`.
        let input = [0.5_f32, 0.5, 0.5];
        let tolerance = default_tolerance_for(cell.op, cell.dtype);
        let decision = fixture_from_consensus(cell, &input, &outputs, tolerance);
        match decision {
            ConsensusDecision::Fixture(f) => fixtures.push(f),
            ConsensusDecision::NoConsensus(reason) => {
                let previews = outputs
                    .iter()
                    .map(|o| {
                        let head: Vec<f32> =
                            o.output.iter().take(8).copied().collect();
                        (o.backend_label.clone(), o.kernel_source.clone(), head)
                    })
                    .collect();
                review.push(ReviewEntry { cell, reason, previews });
            }
        }
    }
    let fixtures_written = fixtures.len();
    let grouped = group_fixtures_for_emission(fixtures);
    for (rel_path, file) in &grouped {
        let written = write_fixture_file(out_dir, rel_path, file)?;
        eprintln!("fuel-capture-fixtures: wrote {}", written.display());
    }
    let cells_skipped = review.entries.len();
    if !review.is_empty() {
        let review_path = review_file
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| out_dir.join("v1").join("REVIEW.md"));
        if let Some(parent) = review_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&review_path, review.to_text())?;
        eprintln!(
            "fuel-capture-fixtures: review report → {}",
            review_path.display(),
        );
    }
    Ok(RunSummary { fixtures_written, cells_skipped })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Help flag and bare invocation both produce Cli::Help.
    #[test]
    fn parse_args_help_or_empty() {
        assert!(matches!(parse_args(&[]).unwrap(), Cli::Help));
        assert!(matches!(parse_args(&["--help".to_string()]).unwrap(), Cli::Help));
        assert!(matches!(parse_args(&["-h".to_string()]).unwrap(), Cli::Help));
    }

    /// `--list-cells` parses to ListCells.
    #[test]
    fn parse_args_list_cells() {
        assert!(matches!(
            parse_args(&["--list-cells".to_string()]).unwrap(),
            Cli::ListCells,
        ));
    }

    /// `--mock` with no `--out-dir` defaults to `./fixtures`.
    #[test]
    fn parse_args_mock_default_out_dir() {
        let cli = parse_args(&["--mock".to_string()]).unwrap();
        match cli {
            Cli::Mock { out_dir, review_file } => {
                assert_eq!(out_dir, PathBuf::from("fixtures"));
                assert!(review_file.is_none());
            }
            other => panic!("expected Mock, got {other:?}"),
        }
    }

    /// `--mock --out-dir <PATH>` picks up the path.
    #[test]
    fn parse_args_mock_with_out_dir() {
        let cli = parse_args(&[
            "--mock".to_string(),
            "--out-dir".to_string(),
            "/tmp/foo".to_string(),
        ]).unwrap();
        match cli {
            Cli::Mock { out_dir, .. } => {
                assert_eq!(out_dir, PathBuf::from("/tmp/foo"));
            }
            other => panic!("expected Mock, got {other:?}"),
        }
    }

    /// Missing value after `--out-dir` is an error.
    #[test]
    fn parse_args_out_dir_requires_value() {
        let err = parse_args(&["--mock".to_string(), "--out-dir".to_string()]).unwrap_err();
        assert!(err.contains("--out-dir"));
    }

    /// Unknown flag is an error.
    #[test]
    fn parse_args_unknown_flag_errors() {
        let err = parse_args(&["--unknown".to_string()]).unwrap_err();
        assert!(err.contains("unknown argument"));
    }

    /// End-to-end mock run writes fixture files and produces no
    /// review entries (mock backends agree by construction).
    #[test]
    fn mock_run_writes_fixtures_no_review() {
        let tmp = std::env::temp_dir().join(format!(
            "fuel-capture-fixtures-mock-{}",
            std::process::id(),
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        let summary = run_mock(&tmp, None).expect("run_mock");
        // Mock cells all agree → every cell produces a fixture.
        // Grouped into 4 files (one per op) under v1/f32/.
        assert!(summary.fixtures_written > 0);
        assert_eq!(summary.cells_skipped, 0);
        let v1 = tmp.join("v1").join("f32");
        let matmul = v1.join("matmul.json");
        assert!(matmul.exists(), "{} should exist", matmul.display());
        let raw = std::fs::read_to_string(&matmul).expect("read matmul.json");
        // Spot-check the file parses as JSON.
        let v: serde_json::Value = serde_json::from_str(&raw).expect("parse");
        assert!(v.get("version").is_some());
        assert!(v.get("fixtures").is_some());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Mock outputs for a given cell are deterministic across
    /// invocations (the capture pipeline is reproducible).
    #[test]
    fn mock_outputs_are_deterministic() {
        let cell = representative_capture_matrix()[0];
        let a = mock_outputs_for(cell);
        let b = mock_outputs_for(cell);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.output, y.output);
            assert_eq!(x.backend_label, y.backend_label);
        }
    }
}
