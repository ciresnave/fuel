//! Capture pipeline for correctness fixtures.
//!
//! This module provides the building blocks for the
//! `fuel-capture-fixtures` binary: deterministic input generation,
//! pairwise consensus clustering across multi-backend measurements,
//! consensus-median fixture selection, and grouped JSON output via
//! [`FixtureFile`].
//!
//! ## Why a module (not the binary directly)
//!
//! The architectural decision is to keep all *capture logic*
//! testable without invoking the Judge. The binary thin-shells
//! these helpers and is the only thing that needs to integrate
//! with the live Judge. Unit tests in this module exercise
//! consensus selection, fixture-file writing, and seed-determinism
//! using mock outputs — no hardware required.
//!
//! ## Pairwise consensus (mirrored from `fuel-core::judge`)
//!
//! We re-implement [`compute_pairwise_consensus`] inline rather
//! than depend on `fuel-core` because the latter pulls in the
//! entire backend stack — a heavy dependency for a tool whose
//! offline review path needs zero backends. The algorithm matches
//! Judge's (greedy mutual-agreement clustering, `1e-3` epsilon by
//! default).
//!
//! ## Capture flow
//!
//! 1. For each `(op, dtype, size_class)` cell in the capture matrix:
//!    a. Build a deterministic input from `(op, dtype, size_class,
//!       input_seed)` via [`deterministic_f32_input`].
//!    b. Hash the input bytes via [`hash_input_bytes`].
//!    c. Collect each backend's output as a [`MeasuredOutput`].
//!    d. Run [`compute_pairwise_consensus`] over the outputs.
//!    e. If majority consensus reached → [`fixture_from_consensus`].
//!    f. Else → log to stderr, write outliers to a review report.
//! 2. Group fixtures by `(op, dtype)` and write each group via
//!    [`write_fixture_file`].

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};

use fuel_core_types::dispatch::{OpKind, SizeClass};
use fuel_core_types::DType;

use crate::{CorrectnessFixture, FixtureFile, ToleranceBand, FIXTURE_FILE_VERSION};

/// Default epsilon for pairwise consensus clustering — matches
/// `fuel-core::judge::CONSENSUS_EPSILON`. Two outputs are in
/// consensus iff their max element-wise relative error is below
/// this threshold.
pub const CAPTURE_CONSENSUS_EPSILON: f32 = 1e-3;

/// One backend's measurement of a single `(op, dtype, size_class)`
/// cell. Constructed by the capture binary after running the Judge's
/// measurement path; used as input to consensus + fixture
/// generation.
///
/// The `backend_label` is purely diagnostic — it appears in the
/// human-review report when consensus fails. Capture itself groups
/// only on `(op, dtype, size_class)`; backend identity is irrelevant
/// to the resulting fixture.
#[derive(Debug, Clone)]
pub struct MeasuredOutput {
    /// Backend identifier for diagnostics (e.g. `"cpu:portable"`,
    /// `"cpu:mkl"`, `"cuda:0"`). Combined with `kernel_source` it
    /// uniquely identifies the kernel that produced `output`.
    pub backend_label: String,
    /// Kernel-source tag from the binding-table entry — distinguishes
    /// sibling kernels (AOCL vs MKL, cuBLAS vs CUTLASS). Empty for
    /// single-alternative cells.
    pub kernel_source: String,
    /// Raw f32 output bytes (reinterpret via `bytemuck::cast_slice`
    /// or `f32::from_le_bytes`). f32 is the only dtype Judge
    /// profiles today; extending to f16/bf16/f64 is a per-arm add.
    pub output: Vec<f32>,
}

/// A single cell in the capture matrix — the addressable unit at
/// which we collect outputs, compute consensus, and emit fixtures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CaptureCell {
    pub op: OpKind,
    pub dtype: DType,
    pub size_class: SizeClass,
    /// Deterministic input seed. The capture tool fixes this per
    /// cell so two runs on the same hardware regenerate byte-
    /// identical inputs (and the resulting fixture is reproducible).
    pub input_seed: u64,
}

/// Conservative capture matrix used when the live `PROFILED_OPS`
/// list is unavailable (e.g. when the capture binary is built
/// without the heavy `fuel-core` dep). Covers four ops × three
/// size classes × f32 — enough to validate the pipeline end-to-end.
///
/// **Per-op size ladders** mirror `fuel-core::judge::Judge::size_plan`
/// exactly. MatMul uses `(m, n, k) ∈ {64³, 256³, 1024³}` → output
/// element counts {4096, 65536, 1048576} → `SizeClass(12, 16, 20)`;
/// Elementwise (Add/Mul) uses `n ∈ {1<<10, 1<<16, 1<<20}` →
/// `SizeClass(10, 16, 20)`. The SoftmaxLastDim cell is included for
/// pipeline shape coverage but is not in Judge's `PROFILED_OPS`
/// today — it inherits the elementwise ladder shape as a placeholder
/// (will follow Judge once Judge profiles fused composites).
///
/// Future work wires the binary directly against the live size_plan
/// so the two stay in lockstep without a manual mirror.
pub fn representative_capture_matrix() -> Vec<CaptureCell> {
    // Each entry: (op, per-op output element counts). The size_class
    // is `SizeClass::from_elem_count(elem_count)` — same bucketing
    // Judge uses for `ProfileEntry::size_class`, so capture cells
    // round-trip into the dispatch table the Judge would emit.
    //
    // MatMul: output = m * n for (64,64,64), (256,256,256),
    // (1024,1024,1024) → 4096, 65536, 1048576.
    // Elementwise: n for 1<<10, 1<<16, 1<<20 → 1024, 65536, 1048576.
    let plans: &[(OpKind, &[usize])] = &[
        (OpKind::MatMul,          &[64 * 64, 256 * 256, 1024 * 1024]),
        (OpKind::AddElementwise,  &[1 << 10, 1 << 16, 1 << 20]),
        (OpKind::MulElementwise,  &[1 << 10, 1 << 16, 1 << 20]),
        (OpKind::SoftmaxLastDim,  &[1 << 10, 1 << 16, 1 << 20]),
    ];
    let mut out = Vec::new();
    for (op, sizes) in plans {
        for &n in *sizes {
            let sc = SizeClass::from_elem_count(n);
            out.push(CaptureCell {
                op: *op,
                dtype: DType::F32,
                size_class: sc,
                // Per-cell seeds are derived from the cell identity
                // so two captures on the same hardware regenerate
                // bit-identical input bytes. Mixing in the op /
                // dtype / size_class avoids cross-cell input
                // collisions (which would make every fixture share
                // the same input_hash — confusing for reviewers).
                input_seed: derive_seed(*op, DType::F32, sc),
            });
        }
    }
    out
}

/// Deterministic per-cell seed. We avoid `Hash::hash` over the
/// cell because `DefaultHasher` isn't stable across Rust versions
/// (per std docs) — that would make fixtures captured on one
/// toolchain irreproducible on another. Instead, derive the seed
/// from a stable u64-mixed combination of the op's `as_str()`
/// (stable, public API), the dtype byte size, and the size class.
fn derive_seed(op: OpKind, dtype: DType, sc: SizeClass) -> u64 {
    // SplitMix-style mix of stable inputs. `op.as_str()` is
    // committed-stable per `fuel_core_types::dispatch::OpKind::as_str`
    // (used in profile-report serialization — changing it is a
    // breaking change). Byte size of dtype is stable across releases.
    let mut acc: u64 = 0xcafe_babe_dead_beef;
    for b in op.as_str().as_bytes() {
        acc = acc.wrapping_mul(0x100000001b3).wrapping_add(*b as u64);
    }
    acc = acc.wrapping_mul(0x100000001b3).wrapping_add(dtype.size_in_bytes() as u64);
    acc = acc.wrapping_mul(0x100000001b3).wrapping_add(sc.0 as u64);
    acc
}

/// Generate deterministic f32 input data for an `(op, element_count)`
/// pair, mirroring the formulas in `fuel-core::judge::build_input_graph`
/// so a fixture's `expected_output` reproduces what Judge measured
/// against the same backend.
///
/// **Why mirror Judge instead of using rand?** Fixtures distribute
/// `(input_seed, expected_output)` pairs. The validator regenerates
/// the input from `input_seed` and asks the local backend to compute
/// the op; that output must match `expected_output` within the
/// fixture's `ToleranceBand`. If capture uses uniform-random inputs
/// but Judge uses sin/cos, the two regenerate divergent bytes and
/// no fixture would ever validate. The cross-reference site is
/// `fuel-core/src/judge/mod.rs::build_input_graph` (see also
/// `unary_input` and `binary_inputs` in that file).
///
/// **Per-op shape:**
/// - `MatMul`: returns the concatenated `[a_data, b_data]` buffer
///   where `a` is `sin(i * 1.3e-3)` over `m * k` elements and `b` is
///   `cos(i * 1.7e-3)` over `k * n` elements. The capture caller
///   slices the buffer back into the two operands before measuring.
///   For square `m == n == k` the buffer length is `2 * m * k`.
/// - Binary elementwise (Add/Sub/Mul/Div/...): returns concatenated
///   `[a, b]` where `a = sin(i * 2.1e-3)` and `b = cos(i * 1.9e-3)`
///   (length `2 * n`). Div/Pow/Rem domain-shifting matches Judge.
/// - Unary elementwise and reductions: returns `sin(i * 2.1e-3)`
///   over `n` elements. Sqrt/Log/Recip/Rsqrt's `+ 1.5` shift matches
///   Judge.
/// - Other ops (e.g. SoftmaxLastDim, not in Judge's PROFILED_OPS):
///   falls back to the unary sin formula. When Judge adds them, this
///   helper grows a matching arm.
pub fn deterministic_f32_input(op: OpKind, element_count: usize) -> Vec<f32> {
    if is_binary_op(op) {
        binary_inputs_concatenated(op, element_count)
    } else if is_matmul_op(op) {
        // Caller passes element_count = m * k + k * n. For the
        // common square case (m == n == k) this is 2 * m * k. We
        // can't reconstruct m / n / k from the count alone — the
        // matrix above is the canonical shape source — but we can
        // emit `a_data` (sin) for the first half and `b_data` (cos)
        // for the second half, which is what Judge does element-wise.
        let half = element_count / 2;
        let mut out = Vec::with_capacity(element_count);
        for i in 0..half {
            out.push((i as f32 * 1.3e-3).sin());
        }
        for i in 0..(element_count - half) {
            out.push((i as f32 * 1.7e-3).cos());
        }
        out
    } else {
        unary_input_with_shift(op, element_count)
    }
}

/// Mirrors `fuel-core::judge::is_binary_elementwise`.
fn is_binary_op(op: OpKind) -> bool {
    matches!(
        op,
        OpKind::AddElementwise
        | OpKind::SubElementwise
        | OpKind::MulElementwise
        | OpKind::DivElementwise
        | OpKind::MaximumElementwise
        | OpKind::MinimumElementwise
        | OpKind::PowElementwise
        | OpKind::RemElementwise,
    )
}

fn is_matmul_op(op: OpKind) -> bool {
    matches!(op, OpKind::MatMul)
}

/// Mirrors `fuel-core::judge::binary_inputs` but returns a single
/// concatenated `[a, b]` buffer (the capture pipeline ships one
/// buffer per fixture; the validator splits it on the way back).
fn binary_inputs_concatenated(op: OpKind, total_elem_count: usize) -> Vec<f32> {
    let n = total_elem_count / 2;
    let mut a: Vec<f32> = (0..n).map(|i| ((i as f32) * 2.1e-3).sin()).collect();
    let mut b: Vec<f32> = (0..(total_elem_count - n))
        .map(|i| ((i as f32) * 1.9e-3).cos())
        .collect();
    if matches!(op, OpKind::DivElementwise) {
        for x in &mut b { *x += 1.5; }
    }
    if matches!(op, OpKind::PowElementwise) {
        // Both inputs must be positive (Judge's domain).
        for x in &mut a { *x += 1.5; }
        for x in &mut b { *x += 1.5; }
    }
    if matches!(op, OpKind::RemElementwise) {
        // Divisor away from zero.
        for x in &mut b { *x += 1.5; }
    }
    let mut out = a;
    out.extend(b);
    out
}

/// Mirrors `fuel-core::judge::unary_input` with the same `+1.5` shift
/// for sqrt/log/recip/rsqrt that Judge applies.
fn unary_input_with_shift(op: OpKind, n: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..n).map(|i| ((i as f32) * 2.1e-3).sin()).collect();
    let needs_nonzero = matches!(
        op,
        OpKind::SqrtElementwise
        | OpKind::LogElementwise
        | OpKind::RecipElementwise
        | OpKind::RsqrtElementwise,
    );
    if needs_nonzero {
        raw.into_iter().map(|x| x + 1.5).collect()
    } else {
        raw
    }
}

/// Hash the byte representation of an input. `DefaultHasher` isn't
/// committed-stable across Rust versions, but the hash is only
/// used as a *sanity check* (regenerate same bytes → same hash) —
/// the input itself is the ground truth. If/when we move to a
/// committed hash, [`CorrectnessFixture::input_hash`] gains a
/// version tag.
pub fn hash_input_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/// Convenience: hash an f32 slice's little-endian byte
/// representation. Mirrors the on-disk format of f32 fixture
/// payloads, so the same hash applies whether you have the f32s
/// in memory or the bytes after deserialization.
pub fn hash_f32_input(values: &[f32]) -> u64 {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    hash_input_bytes(&bytes)
}

/// Compute the largest mutually-close cluster across `outputs`.
/// Returns indices into `outputs` (sorted ascending) of the
/// consensus members.
///
/// Mirrors `fuel-core::judge::compute_pairwise_consensus`. Two
/// outputs are in consensus iff their max element-wise relative
/// error is below `epsilon`. Greedy expansion finds the largest
/// such cluster; ties broken by lowest starting index.
///
/// Edge cases:
/// - Empty `outputs`: empty consensus.
/// - Single output: trivial consensus `[0]`. With a single
///   backend there's no peer to disagree with; the capture binary
///   treats single-backend cells as "no consensus possible —
///   skip" elsewhere.
/// - All outputs agree: consensus is `[0, 1, ..., n-1]`.
/// - Outlier scenario: consensus is the agreeing subset.
pub fn compute_pairwise_consensus(outputs: &[MeasuredOutput], epsilon: f32) -> Vec<usize> {
    let n = outputs.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }
    let mut agree = vec![vec![false; n]; n];
    for i in 0..n {
        agree[i][i] = true;
        for j in (i + 1)..n {
            let close = max_rel_err(&outputs[i].output, &outputs[j].output) < epsilon;
            agree[i][j] = close;
            agree[j][i] = close;
        }
    }
    let mut best: Vec<usize> = vec![0];
    for i in 0..n {
        let mut cluster = vec![i];
        for j in 0..n {
            if j == i { continue; }
            if cluster.iter().all(|&k| agree[j][k]) {
                cluster.push(j);
            }
        }
        cluster.sort_unstable();
        if cluster.len() > best.len() {
            best = cluster;
        }
    }
    best
}

/// Max element-wise relative error between two same-length f32
/// vectors. NaN/Inf in either vector → INFINITY (never "in
/// consensus"). Mirrors `fuel-core::judge::max_rel_err`.
fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    let mut worst = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return f32::INFINITY;
        }
        let denom = x.abs().max(y.abs()).max(f32::MIN_POSITIVE);
        let rel = (x - y).abs() / denom;
        if rel > worst { worst = rel; }
    }
    worst
}

/// Decision returned by [`fixture_from_consensus`]: either a
/// fixture is produced (consensus reached) or the cell is flagged
/// for human review (no consensus, or single-backend).
#[derive(Debug)]
pub enum ConsensusDecision {
    /// Consensus achieved — fixture is the median of the consensus
    /// cluster.
    Fixture(CorrectnessFixture),
    /// No consensus and / or fewer than 2 backends measured. The
    /// caller logs a human-review report and skips emitting a
    /// fixture.
    NoConsensus(NoConsensusReason),
}

/// Why a cell didn't produce a fixture. Surfaced in the human-
/// review report.
#[derive(Debug, Clone)]
pub enum NoConsensusReason {
    /// Fewer than 2 backends measured this cell. Single-backend
    /// systems can't compute meaningful consensus; the fixture
    /// would just be "whatever the lone backend produced" without
    /// any independent validation — which is exactly what fixtures
    /// exist to avoid. Skip.
    InsufficientPeers { backend_count: usize },
    /// Two or more backends measured but no strict-majority cluster
    /// (cluster size × 2 ≤ N). For N=2 a strict majority means both
    /// agree; for N=4 it means at least three agree; for N=3 it
    /// means at least two. Human review needed.
    NoMajority {
        backend_count: usize,
        largest_cluster_size: usize,
    },
}

impl std::fmt::Display for NoConsensusReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientPeers { backend_count } => write!(
                f,
                "only {backend_count} backend(s) measured — need >=2 for consensus",
            ),
            Self::NoMajority { backend_count, largest_cluster_size } => write!(
                f,
                "{backend_count} backends measured but largest agreeing cluster \
                 is only {largest_cluster_size} (need majority for fixture)",
            ),
        }
    }
}

/// Build a [`CorrectnessFixture`] from a cell's measurements iff
/// the pairwise-consensus cluster is a majority of `outputs`.
///
/// The consensus median is the per-element median of the
/// consensus cluster's outputs — this is the value we ship.
/// Median (not mean) because median ignores outliers within the
/// consensus group (which can happen when 3 backends agree to
/// 1e-4 but one of them is slightly higher than the other two on
/// most elements).
pub fn fixture_from_consensus(
    cell: CaptureCell,
    input: &[f32],
    outputs: &[MeasuredOutput],
    tolerance: ToleranceBand,
) -> ConsensusDecision {
    let n = outputs.len();
    if n < 2 {
        return ConsensusDecision::NoConsensus(
            NoConsensusReason::InsufficientPeers { backend_count: n },
        );
    }
    let consensus = compute_pairwise_consensus(outputs, CAPTURE_CONSENSUS_EPSILON);
    // Strict majority: `consensus.len() * 2 > n` (equivalently
    // `consensus.len() > n / 2`). The earlier `n.div_ceil(2)`
    // bound was wrong for even N — for N=2 it admitted a single
    // disagreer's output as the fixture, and for N=4 it admitted
    // a 2-vs-2 split. A strict-majority fixture means at least
    // ⌊N/2⌋ + 1 backends agree; for N=2 that's both, for N=4
    // that's three, for N=3 that's two (i.e. still permissive on
    // odd N where a tie is impossible).
    if consensus.len() * 2 <= n {
        return ConsensusDecision::NoConsensus(NoConsensusReason::NoMajority {
            backend_count: n,
            largest_cluster_size: consensus.len(),
        });
    }
    let consensus_outputs: Vec<&[f32]> = consensus
        .iter()
        .map(|&i| outputs[i].output.as_slice())
        .collect();
    let median = elementwise_median_f32(&consensus_outputs);
    let mut bytes = Vec::with_capacity(median.len() * 4);
    for v in &median {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let input_hash = hash_f32_input(input);
    ConsensusDecision::Fixture(CorrectnessFixture {
        op: cell.op,
        dtype: cell.dtype,
        size_class: cell.size_class,
        input_seed: cell.input_seed,
        input_hash,
        expected_output: bytes,
        output_element_count: median.len(),
        tolerance,
    })
}

/// Per-element median of N same-length f32 slices. Returns an empty
/// vec if `outputs` is empty or any output has a mismatched length;
/// production callers should pre-validate (consensus cluster output
/// lengths should match by construction).
fn elementwise_median_f32(outputs: &[&[f32]]) -> Vec<f32> {
    let n = outputs.len();
    if n == 0 {
        return Vec::new();
    }
    let len = outputs[0].len();
    if outputs.iter().any(|o| o.len() != len) {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(len);
    let mut scratch: Vec<f32> = Vec::with_capacity(n);
    for i in 0..len {
        scratch.clear();
        for o in outputs {
            scratch.push(o[i]);
        }
        scratch.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Lower median for even counts — deterministic, matches
        // numpy's `kind='lower'`. Avoids averaging two near-equal
        // f32s which would introduce a quarter-ulp of synthetic
        // rounding into the fixture.
        result.push(scratch[(n - 1) / 2]);
    }
    result
}

/// Group fixtures by `(dtype, op)` for file emission. Returns a
/// map keyed by the relative output path (e.g.
/// `v1/f32/matmul.json`) → its FixtureFile. The caller renders the
/// map to disk via [`write_fixture_file`] (or any other I/O surface).
pub fn group_fixtures_for_emission(
    fixtures: Vec<CorrectnessFixture>,
) -> HashMap<PathBuf, FixtureFile> {
    let mut by_path: HashMap<PathBuf, Vec<CorrectnessFixture>> = HashMap::new();
    for f in fixtures {
        let path = PathBuf::from(format!(
            "v1/{dtype}/{op}.json",
            dtype = dtype_dir_name(f.dtype),
            op = f.op.as_str(),
        ));
        by_path.entry(path).or_default().push(f);
    }
    by_path
        .into_iter()
        .map(|(path, mut fs)| {
            // Stable order: by size_class then by input_seed. The
            // file-on-disk diff stays sane when a capture run adds
            // or modifies a single cell.
            fs.sort_by_key(|f| (f.size_class.0, f.input_seed));
            (path, FixtureFile {
                version: FIXTURE_FILE_VERSION,
                fixtures: fs,
            })
        })
        .collect()
}

/// Canonical directory name for a dtype in the on-disk fixture
/// layout. Matches the dtype's lowercase `Display` shape; new
/// dtypes get an arm here as they become Judge-profiled.
fn dtype_dir_name(d: DType) -> &'static str {
    match d {
        DType::U8 => "u8",
        DType::I8 => "i8",
        DType::U32 => "u32",
        DType::I16 => "i16",
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::F8E4M3 => "f8e4m3",
        DType::F6E2M3 => "f6e2m3",
        DType::F6E3M2 => "f6e3m2",
        DType::F4 => "f4",
        DType::F8E8M0 => "f8e8m0",
    }
}

/// Write a [`FixtureFile`] to `<root>/<rel_path>` as pretty-printed
/// JSON. Creates parent directories as needed. Returns the full
/// path written.
pub fn write_fixture_file(
    root: &Path,
    rel_path: &Path,
    file: &FixtureFile,
) -> std::io::Result<PathBuf> {
    let full = root.join(rel_path);
    if let Some(parent) = full.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(&full, json)?;
    Ok(full)
}

/// One line in the human-review report. Aggregated by
/// [`ReviewReport`] for cells where consensus failed.
#[derive(Debug, Clone)]
pub struct ReviewEntry {
    pub cell: CaptureCell,
    pub reason: NoConsensusReason,
    /// (`backend_label`, `kernel_source`, first 8 output elements)
    /// — enough for a human reviewer to see at a glance where
    /// outputs diverged. Full byte payloads sit in the per-run
    /// log files (not emitted by the capture binary).
    pub previews: Vec<(String, String, Vec<f32>)>,
}

/// Aggregated human-review report. Written next to the fixture
/// tree as `<root>/v1/REVIEW.json` (or wherever the caller prefers).
#[derive(Debug, Default)]
pub struct ReviewReport {
    pub entries: Vec<ReviewEntry>,
}

impl ReviewReport {
    pub fn new() -> Self { Self::default() }
    pub fn push(&mut self, entry: ReviewEntry) {
        self.entries.push(entry);
    }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    /// Render the report as a human-readable string for stderr or
    /// a `.txt` log. Stable ordering: by op, dtype, size_class.
    pub fn to_text(&self) -> String {
        let mut entries = self.entries.clone();
        entries.sort_by_key(|e| (e.cell.op.as_str(), e.cell.dtype.size_in_bytes(), e.cell.size_class.0));
        let mut out = String::new();
        out.push_str("# Correctness Capture — Human Review Required\n\n");
        if entries.is_empty() {
            out.push_str("(no entries — all cells reached consensus)\n");
            return out;
        }
        for entry in &entries {
            out.push_str(&format!(
                "## {op} / {dtype:?} / size_class={sc}\n",
                op = entry.cell.op.as_str(),
                dtype = entry.cell.dtype,
                sc = entry.cell.size_class.0,
            ));
            out.push_str(&format!("- input_seed: {}\n", entry.cell.input_seed));
            out.push_str(&format!("- reason: {}\n", entry.reason));
            for (backend, ks, preview) in &entry.previews {
                let ks_tag = if ks.is_empty() { "" } else { ks.as_str() };
                out.push_str(&format!(
                    "  - {backend} ({ks_tag}): {:?}\n",
                    preview,
                ));
            }
            out.push('\n');
        }
        out
    }
}

/// Tolerance band selection for a captured op. IEEE-bit-stable
/// elementwise ops (`AddElementwise` / `SubElementwise` /
/// `MulElementwise`) ship `F32_STRICT`; every other op ships the
/// general `F32_DEFAULT`. Wired as a separate function so callers
/// can extend without rewriting the caller chain.
///
/// **Why these three?** Per `PrecisionGuarantee::BitStable`,
/// IEEE-754 single-precision add/sub/mul are deterministic across
/// any backend that respects IEEE round-to-nearest-even (every
/// backend Fuel ships against). The other elementwise ops admit
/// implementation latitude (transcendentals especially) and need
/// the wider default tolerance.
pub fn default_tolerance_for(op: OpKind, dtype: DType) -> ToleranceBand {
    let _ = dtype; // future: per-dtype defaults
    match op {
        OpKind::AddElementwise
        | OpKind::SubElementwise
        | OpKind::MulElementwise => ToleranceBand::F32_STRICT,
        _ => ToleranceBand::F32_DEFAULT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_output(label: &str, values: Vec<f32>) -> MeasuredOutput {
        MeasuredOutput {
            backend_label: label.to_string(),
            kernel_source: String::new(),
            output: values,
        }
    }

    /// Two backends with byte-identical outputs cluster together.
    #[test]
    fn consensus_two_agreeing_backends() {
        let outs = vec![
            mock_output("cpu", vec![1.0, 2.0, 3.0]),
            mock_output("cuda:0", vec![1.0, 2.0, 3.0]),
        ];
        let consensus = compute_pairwise_consensus(&outs, CAPTURE_CONSENSUS_EPSILON);
        assert_eq!(consensus, vec![0, 1]);
    }

    /// Three backends, one outlier — consensus excludes the outlier.
    #[test]
    fn consensus_excludes_outlier() {
        let outs = vec![
            mock_output("cpu", vec![1.0, 2.0, 3.0]),
            mock_output("cuda:0", vec![1.0, 2.0, 3.0]),
            mock_output("buggy", vec![10.0, 20.0, 30.0]),
        ];
        let consensus = compute_pairwise_consensus(&outs, CAPTURE_CONSENSUS_EPSILON);
        assert_eq!(consensus, vec![0, 1]);
    }

    /// Single backend → trivial consensus of [0]. The
    /// fixture-from-consensus layer is what rejects single-backend
    /// cells; bare consensus computation still returns [0].
    #[test]
    fn consensus_single_backend_is_trivial() {
        let outs = vec![mock_output("cpu", vec![1.0, 2.0, 3.0])];
        let consensus = compute_pairwise_consensus(&outs, CAPTURE_CONSENSUS_EPSILON);
        assert_eq!(consensus, vec![0]);
    }

    /// Empty input → empty consensus.
    #[test]
    fn consensus_empty_is_empty() {
        let outs: Vec<MeasuredOutput> = vec![];
        let consensus = compute_pairwise_consensus(&outs, CAPTURE_CONSENSUS_EPSILON);
        assert!(consensus.is_empty());
    }

    /// Two disagreeing backends → consensus is one of them
    /// (deterministically the first by index). The
    /// fixture-from-consensus layer flags this as NoMajority because
    /// 1 < ⌈2/2⌉ = 1 isn't a strict majority — but the algorithm
    /// returns it.
    #[test]
    fn consensus_two_disagreeing_picks_first() {
        let outs = vec![
            mock_output("cpu", vec![1.0, 2.0, 3.0]),
            mock_output("buggy", vec![100.0, 200.0, 300.0]),
        ];
        let consensus = compute_pairwise_consensus(&outs, CAPTURE_CONSENSUS_EPSILON);
        // Singleton — but which one? The greedy algorithm picks
        // the lowest-index singleton when no peer agrees.
        assert_eq!(consensus.len(), 1);
        assert_eq!(consensus[0], 0);
    }

    /// Single-backend cell → NoConsensus(InsufficientPeers).
    #[test]
    fn fixture_from_consensus_rejects_single_backend() {
        let cell = CaptureCell {
            op: OpKind::MatMul,
            dtype: DType::F32,
            size_class: SizeClass(10),
            input_seed: 42,
        };
        let outs = vec![mock_output("cpu", vec![1.0, 2.0])];
        let input = vec![0.5_f32, 0.5];
        let decision = fixture_from_consensus(cell, &input, &outs, ToleranceBand::F32_DEFAULT);
        match decision {
            ConsensusDecision::NoConsensus(NoConsensusReason::InsufficientPeers { backend_count }) => {
                assert_eq!(backend_count, 1);
            }
            other => panic!("expected InsufficientPeers, got {other:?}"),
        }
    }

    /// Two disagreeing backends → NoConsensus(NoMajority). Strict
    /// majority means consensus.len() * 2 > n; for N=2 a singleton
    /// (1 * 2 == 2) is NOT a strict majority. The pre-fix behavior
    /// (admit the first backend's output as the fixture) was wrong
    /// — a lone backend's word isn't a fixture, it's an outlier.
    #[test]
    fn fixture_from_consensus_two_backends_disagree_no_majority() {
        let cell = CaptureCell {
            op: OpKind::AddElementwise,
            dtype: DType::F32,
            size_class: SizeClass(10),
            input_seed: 7,
        };
        let outs = vec![
            mock_output("cpu", vec![1.0, 2.0]),
            mock_output("buggy", vec![100.0, 200.0]),
        ];
        let input = vec![0.5_f32, 0.5];
        let decision = fixture_from_consensus(cell, &input, &outs, ToleranceBand::F32_DEFAULT);
        match decision {
            ConsensusDecision::NoConsensus(NoConsensusReason::NoMajority {
                backend_count,
                largest_cluster_size,
            }) => {
                assert_eq!(backend_count, 2);
                assert_eq!(largest_cluster_size, 1);
            }
            other => panic!("expected NoMajority, got {other:?}"),
        }
    }

    /// Four backends, two-vs-two split → NoConsensus(NoMajority).
    /// The earlier `n.div_ceil(2)` threshold admitted this case as
    /// a fixture (cluster 2 == threshold 2); strict majority
    /// (cluster.len() * 2 > N) correctly rejects it: 2 * 2 == 4,
    /// not > 4. This is the cell-loss case the pre-fix capture
    /// would silently emit.
    #[test]
    fn fixture_from_consensus_four_backends_two_two_split_no_majority() {
        let cell = CaptureCell {
            op: OpKind::AddElementwise,
            dtype: DType::F32,
            size_class: SizeClass(10),
            input_seed: 19,
        };
        let outs = vec![
            mock_output("a", vec![1.0, 2.0, 3.0]),
            mock_output("b", vec![1.0, 2.0, 3.0]),
            mock_output("c", vec![10.0, 20.0, 30.0]),
            mock_output("d", vec![10.0, 20.0, 30.0]),
        ];
        let input = vec![0.5_f32, 0.5, 0.5];
        let decision = fixture_from_consensus(cell, &input, &outs, ToleranceBand::F32_DEFAULT);
        match decision {
            ConsensusDecision::NoConsensus(NoConsensusReason::NoMajority {
                backend_count,
                largest_cluster_size,
            }) => {
                assert_eq!(backend_count, 4);
                assert_eq!(largest_cluster_size, 2);
            }
            other => panic!("expected NoMajority, got {other:?}"),
        }
    }

    /// Three-backend cell with one outlier → fixture produced
    /// from the agreeing pair's median.
    #[test]
    fn fixture_from_consensus_three_backends_one_outlier() {
        let cell = CaptureCell {
            op: OpKind::MatMul,
            dtype: DType::F32,
            size_class: SizeClass(10),
            input_seed: 11,
        };
        let outs = vec![
            mock_output("cpu", vec![1.0, 2.0, 3.0]),
            mock_output("cuda:0", vec![1.0, 2.0, 3.0]),
            mock_output("buggy", vec![10.0, 20.0, 30.0]),
        ];
        let input = vec![0.5_f32, 0.5, 0.5];
        let decision = fixture_from_consensus(cell, &input, &outs, ToleranceBand::F32_DEFAULT);
        match decision {
            ConsensusDecision::Fixture(f) => {
                let expected: &[f32] = bytemuck::cast_slice(&f.expected_output);
                assert_eq!(expected, &[1.0, 2.0, 3.0]);
                assert_eq!(f.output_element_count, 3);
            }
            other => panic!("expected fixture, got {other:?}"),
        }
    }

    /// Four backends, two-two split → NoMajority (consensus = 2,
    /// majority = 2 — that's exactly the threshold and we PASS).
    /// Five-backend two-three split is a clearer no-majority case.
    #[test]
    fn fixture_from_consensus_five_backends_no_majority() {
        let cell = CaptureCell {
            op: OpKind::MatMul,
            dtype: DType::F32,
            size_class: SizeClass(10),
            input_seed: 13,
        };
        // 2 backends agree on [1,2,3]; 2 agree on [10,20,30]; 1
        // is solo at [100,200,300]. Largest cluster = 2; majority
        // threshold for N=5 is ⌈5/2⌉ = 3. 2 < 3 → NoMajority.
        let outs = vec![
            mock_output("a", vec![1.0, 2.0, 3.0]),
            mock_output("b", vec![1.0, 2.0, 3.0]),
            mock_output("c", vec![10.0, 20.0, 30.0]),
            mock_output("d", vec![10.0, 20.0, 30.0]),
            mock_output("e", vec![100.0, 200.0, 300.0]),
        ];
        let input = vec![0.5_f32, 0.5, 0.5];
        let decision = fixture_from_consensus(cell, &input, &outs, ToleranceBand::F32_DEFAULT);
        match decision {
            ConsensusDecision::NoConsensus(NoConsensusReason::NoMajority { backend_count, largest_cluster_size }) => {
                assert_eq!(backend_count, 5);
                assert_eq!(largest_cluster_size, 2);
            }
            other => panic!("expected NoMajority, got {other:?}"),
        }
    }

    /// Per-element median of an even-length cluster uses the
    /// LOWER median (no synthetic averaging).
    #[test]
    fn elementwise_median_uses_lower_for_even_n() {
        let a = [1.0_f32, 4.0];
        let b = [2.0_f32, 5.0];
        let result = elementwise_median_f32(&[&a, &b]);
        assert_eq!(result, vec![1.0, 4.0]); // lower of (1,2) and (4,5)
    }

    /// Mismatched lengths return empty (defensive fallback).
    #[test]
    fn elementwise_median_returns_empty_on_mismatch() {
        let a = [1.0_f32, 2.0];
        let b = [1.0_f32];
        let result = elementwise_median_f32(&[&a, &b]);
        assert!(result.is_empty());
    }

    /// Same `(op, count)` → byte-identical inputs. This is the
    /// contract that lets validators regenerate captured inputs.
    #[test]
    fn deterministic_input_is_deterministic() {
        let a = deterministic_f32_input(OpKind::SinElementwise, 100);
        let b = deterministic_f32_input(OpKind::SinElementwise, 100);
        assert_eq!(a, b);
    }

    /// Different ops produce different inputs when their formulas
    /// or shifts differ — e.g. SqrtElementwise applies a +1.5
    /// domain shift while SinElementwise doesn't.
    #[test]
    fn different_ops_produce_different_inputs() {
        let unshifted = deterministic_f32_input(OpKind::SinElementwise, 100);
        let shifted = deterministic_f32_input(OpKind::SqrtElementwise, 100);
        assert_ne!(unshifted, shifted);
    }

    /// Hashing is deterministic within a process. (Across-process
    /// stability isn't claimed because `DefaultHasher` doesn't
    /// commit to it.)
    #[test]
    fn hash_is_deterministic_within_process() {
        let a = deterministic_f32_input(OpKind::SinElementwise, 100);
        let b = deterministic_f32_input(OpKind::SinElementwise, 100);
        assert_eq!(hash_f32_input(&a), hash_f32_input(&b));
    }

    /// Capture's `deterministic_f32_input` for a unary op matches
    /// the hand-computed first 8 elements of Judge's `unary_input`
    /// formula `sin(i * 2.1e-3)`. If Judge's formula ever changes,
    /// this test fails — the cross-reference is the canary.
    #[test]
    fn deterministic_input_matches_judge_unary_formula() {
        let actual = deterministic_f32_input(OpKind::SinElementwise, 8);
        let expected: Vec<f32> = (0..8)
            .map(|i| ((i as f32) * 2.1e-3).sin())
            .collect();
        assert_eq!(actual, expected);
    }

    /// Capture's binary inputs concatenate `sin(i * 2.1e-3)` and
    /// `cos(i * 1.9e-3)` (matching Judge's `binary_inputs`). Hand-
    /// compute the first 4 elements of each half and assert.
    #[test]
    fn deterministic_input_matches_judge_binary_formula() {
        let actual = deterministic_f32_input(OpKind::AddElementwise, 8);
        let mut expected: Vec<f32> = (0..4)
            .map(|i| ((i as f32) * 2.1e-3).sin())
            .collect();
        expected.extend((0..4).map(|i| ((i as f32) * 1.9e-3).cos()));
        assert_eq!(actual, expected);
    }

    /// Capture's MatMul inputs concatenate `sin(i * 1.3e-3)` and
    /// `cos(i * 1.7e-3)` (matching Judge's `build_input_graph`
    /// MatMul arm). Hand-compute the first 4 of each half.
    #[test]
    fn deterministic_input_matches_judge_matmul_formula() {
        let actual = deterministic_f32_input(OpKind::MatMul, 8);
        let mut expected: Vec<f32> = (0..4)
            .map(|i| ((i as f32) * 1.3e-3).sin())
            .collect();
        expected.extend((0..4).map(|i| ((i as f32) * 1.7e-3).cos()));
        assert_eq!(actual, expected);
    }

    /// Sqrt's `+1.5` shift puts the unary `[-1, 1]` range into
    /// `[0.5, 2.5]` — matches Judge's `needs_nonzero` arm.
    #[test]
    fn deterministic_input_applies_judge_sqrt_shift() {
        let raw = deterministic_f32_input(OpKind::SinElementwise, 4);
        let shifted = deterministic_f32_input(OpKind::SqrtElementwise, 4);
        for (r, s) in raw.iter().zip(shifted.iter()) {
            assert!((s - (r + 1.5)).abs() < 1e-7, "expected r+1.5={}, got {s}", r + 1.5);
        }
    }

    /// Group → emission paths use `v1/{dtype}/{op}.json`.
    #[test]
    fn group_fixtures_paths_match_v1_layout() {
        let cell_a = CaptureCell {
            op: OpKind::MatMul,
            dtype: DType::F32,
            size_class: SizeClass(10),
            input_seed: 1,
        };
        let cell_b = CaptureCell {
            op: OpKind::AddElementwise,
            dtype: DType::F32,
            size_class: SizeClass(10),
            input_seed: 2,
        };
        let make_fixture = |cell: CaptureCell, vals: Vec<f32>| -> CorrectnessFixture {
            let bytes: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
            CorrectnessFixture {
                op: cell.op,
                dtype: cell.dtype,
                size_class: cell.size_class,
                input_seed: cell.input_seed,
                input_hash: 0,
                expected_output: bytes,
                output_element_count: vals.len(),
                tolerance: ToleranceBand::F32_DEFAULT,
            }
        };
        let fixtures = vec![
            make_fixture(cell_a, vec![1.0]),
            make_fixture(cell_b, vec![2.0]),
        ];
        let grouped = group_fixtures_for_emission(fixtures);
        assert!(grouped.contains_key(&PathBuf::from("v1/f32/matmul.json")));
        assert!(grouped.contains_key(&PathBuf::from("v1/f32/add.json")));
    }

    /// Same (op, dtype) cells with different size_classes group
    /// into ONE file (sorted by size_class within).
    #[test]
    fn group_fixtures_same_op_dtype_one_file() {
        let mk = |sc: u8, seed: u64| -> CorrectnessFixture {
            CorrectnessFixture {
                op: OpKind::MatMul,
                dtype: DType::F32,
                size_class: SizeClass(sc),
                input_seed: seed,
                input_hash: 0,
                expected_output: vec![],
                output_element_count: 0,
                tolerance: ToleranceBand::F32_DEFAULT,
            }
        };
        let fixtures = vec![mk(20, 3), mk(10, 1), mk(16, 2)];
        let grouped = group_fixtures_for_emission(fixtures);
        assert_eq!(grouped.len(), 1);
        let file = grouped.get(&PathBuf::from("v1/f32/matmul.json")).unwrap();
        assert_eq!(file.fixtures.len(), 3);
        // Sorted ascending by size_class.
        let sizes: Vec<u8> = file.fixtures.iter().map(|f| f.size_class.0).collect();
        assert_eq!(sizes, vec![10, 16, 20]);
    }

    /// `write_fixture_file` round-trips through serde — the file
    /// on disk parses back identical.
    #[test]
    fn write_fixture_file_round_trips() {
        let cell = CaptureCell {
            op: OpKind::MatMul,
            dtype: DType::F32,
            size_class: SizeClass(10),
            input_seed: 99,
        };
        let outs = vec![
            mock_output("cpu", vec![1.0, 2.0, 3.0]),
            mock_output("cuda:0", vec![1.0, 2.0, 3.0]),
        ];
        let input = vec![0.5_f32, 0.5, 0.5];
        let decision = fixture_from_consensus(cell, &input, &outs, ToleranceBand::F32_DEFAULT);
        let fixture = match decision {
            ConsensusDecision::Fixture(f) => f,
            other => panic!("expected fixture, got {other:?}"),
        };
        let grouped = group_fixtures_for_emission(vec![fixture.clone()]);
        let (path, file) = grouped.iter().next().unwrap();

        let tmp = std::env::temp_dir().join(format!(
            "fuel-capture-fixtures-test-{}",
            std::process::id(),
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        let written = write_fixture_file(&tmp, path, file).expect("write");
        let raw = std::fs::read_to_string(&written).expect("read");
        let parsed: FixtureFile = serde_json::from_str(&raw).expect("parse");
        assert_eq!(parsed.fixtures.len(), 1);
        assert_eq!(parsed.fixtures[0], fixture);
        // Clean up
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Representative capture matrix has the documented shape.
    #[test]
    fn representative_matrix_shape() {
        let cells = representative_capture_matrix();
        assert_eq!(cells.len(), 4 * 3); // 4 ops × 3 size classes
        // Every cell is f32 (the only profiled dtype today).
        assert!(cells.iter().all(|c| c.dtype == DType::F32));
        // Seeds are distinct across (op, size_class) cells.
        let mut seeds: Vec<u64> = cells.iter().map(|c| c.input_seed).collect();
        seeds.sort_unstable();
        seeds.dedup();
        assert_eq!(seeds.len(), 4 * 3);
    }

    /// Every capture cell's `size_class` mirrors what Judge would
    /// emit for the same op:
    /// - MatMul: outputs 64², 256², 1024² → SizeClass(12, 16, 20)
    /// - Elementwise: 1<<10, 1<<16, 1<<20 → SizeClass(10, 16, 20)
    ///
    /// If Judge's size_plan ever changes, this test fails — the
    /// canary that pulls capture back into lockstep.
    #[test]
    fn representative_matrix_size_classes_match_judge() {
        let cells = representative_capture_matrix();
        let mut by_op: HashMap<OpKind, Vec<u8>> = HashMap::new();
        for c in &cells {
            by_op.entry(c.op).or_default().push(c.size_class.0);
        }
        for v in by_op.values_mut() {
            v.sort_unstable();
        }
        // MatMul: 4096 → 12, 65536 → 16, 1048576 → 20.
        assert_eq!(by_op.get(&OpKind::MatMul), Some(&vec![12u8, 16, 20]));
        // Add/Mul: 1024 → 10, 65536 → 16, 1048576 → 20.
        assert_eq!(by_op.get(&OpKind::AddElementwise), Some(&vec![10u8, 16, 20]));
        assert_eq!(by_op.get(&OpKind::MulElementwise), Some(&vec![10u8, 16, 20]));
        // Softmax inherits the elementwise ladder placeholder until
        // Judge profiles fused composites.
        assert_eq!(by_op.get(&OpKind::SoftmaxLastDim), Some(&vec![10u8, 16, 20]));
    }

    /// Review report renders cleanly even with multiple entries.
    #[test]
    fn review_report_renders_with_entries() {
        let mut report = ReviewReport::new();
        report.push(ReviewEntry {
            cell: CaptureCell {
                op: OpKind::MatMul,
                dtype: DType::F32,
                size_class: SizeClass(10),
                input_seed: 1,
            },
            reason: NoConsensusReason::NoMajority {
                backend_count: 3,
                largest_cluster_size: 1,
            },
            previews: vec![
                ("cpu".to_string(), "".to_string(), vec![1.0, 2.0]),
                ("cuda:0".to_string(), "cublas".to_string(), vec![10.0, 20.0]),
            ],
        });
        let text = report.to_text();
        assert!(text.contains("matmul"));
        assert!(text.contains("NoMajority")
            || text.contains("largest agreeing cluster"));
        assert!(text.contains("cpu"));
        assert!(text.contains("cuda:0"));
    }

    /// Empty review report renders the "no entries" placeholder.
    #[test]
    fn review_report_empty_renders_placeholder() {
        let report = ReviewReport::new();
        let text = report.to_text();
        assert!(text.contains("no entries"));
    }
}
