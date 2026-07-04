//! The Judge — Phase 6b's empirical profiler.
//!
//! For every `(backend, device)` pair the probe report knows about,
//! the Judge walks a matrix of `(op_kind, dtype, size_class)` and
//! measures two things: wall-clock latency and numerical precision
//! relative to the reference backend. Output is a persistable
//! [`ProfileReport`] that the (future) ranked dispatch table indexes
//! at realize time.
//!
//! ## Submodules
//!
//! - [`cache`] — process-wide [`DispatchTable`] cache. Hosts
//!   `cached()`, `populate_dispatch_table()`, `invalidate()`. Was
//!   `fuel_core::dispatch` before the 2026-05-31 dispatch-move; the
//!   name was a misnomer (it caches the Judge's output, not the
//!   binding-table dispatch path), so it now lives under the Judge
//!   module that owns its lifecycle. The cache surface is re-exported
//!   at this module's top level so callers can write
//!   `fuel_core::judge::cached()` directly.
//!
//! # Scope of this revision
//!
//! Ships the types, the runner skeleton, and per-family probe
//! coverage spanning the inference op surface in F32:
//!
//! - **Ops**: see [`PROFILED_OPS`] for the canonical list. Currently:
//!   - dense linear algebra: [`OpKind::MatMul`]
//!   - elementwise binary: Add / Sub / Mul / Div / Maximum / Minimum
//!   - elementwise unary: Neg / Sqr / Sqrt / Exp / Log / Sin / Cos /
//!     Tanh / Sigmoid / Silu / Gelu / Relu / Step
//!   - per-axis reductions: SumReduce / MaxReduce / MinReduce /
//!     MeanReduce (probed as last-dim reduction over `[rows, 64]`)
//!   - reduce-to-broadcast-target: ReduceSumTo / ReduceMaxTo
//!   - parametric one-input: Affine / Clamp / PowI
//!   - 28 OpKind variants total, 84 (op, size) cells per backend
//!     class.
//! - **Dtypes**: `{f32, f16, bf16}` (see [`PROFILED_DTYPES`]). The
//!   dispatch key always carried a `dtype` axis
//!   ([`ProfileEntry::dtype`]); the 2026-07-04 dtype slice made the
//!   *runner* iterate it rather than hard-coding f32. Each profiled
//!   op/size is now measured once per dtype, so the report carries
//!   `3 ×` the cells it did in the f32-only era. Inputs are generated
//!   as deterministic f32 and down-converted to the cell's dtype at
//!   graph-build time; the realizer times the native-dtype kernel and
//!   reads the output back at its native width, converting to f32
//!   only for the cross-backend correctness verdict
//!   ([`crate::factories::LazyRealizer::realize_capture_f32`]). A cell
//!   whose backend has no kernel for the requested dtype is skipped
//!   cleanly (logged), never fatal — CPU currently carries
//!   f16/bf16 kernels for the elementwise/matmul/reduction families
//!   (FKC `[F32,F64,BF16,F16]` fan-out). f64 stays OUT of the matrix:
//!   decode/inference is f16/bf16, and f64's only consumer is
//!   correctness reference paths that don't route through the ranked
//!   dispatch table.
//! - **Backends**: every backend in the [`crate::factories`] registry.
//!   A backend that doesn't implement an op is skipped per cell with
//!   a stderr note (the realize call is wrapped in `catch_unwind`),
//!   not fatal to the run — what makes the Judge safe to expand
//!   ahead of every backend's coverage.
//!
//! Op kinds *not yet profiled* (build_input_graph would `panic!` if
//! the size_plan returned a non-empty ladder for them; the size_plan's
//! catch-all `_ => Vec::new()` arm silently skips):
//!
//! - composition / fused ops: SoftmaxLastDim / RmsNormLastDim /
//!   LayerNormLastDim / Rope / FlashAttn / PagedAttn / FusedLinear.
//!   These are slated for Phase 7.6 fused-op profiling (per ROADMAP),
//!   not the primitive-OpKind sweep this module covers.
//! - shape-rearranging: Cast / IndexSelect / Gather / Concat /
//!   IndexAdd / ScatterAdd / ArgMaxDim / ArgMinDim.
//! - convolutions: Conv2D / ConvTranspose2D.
//! - quantized: QMatMul.
//!
//! Each of those is a separable add: extend [`PROFILED_OPS`], add a
//! `size_plan` arm with a representative size ladder, add a
//! `build_input_graph` arm that constructs the input graph.
//!
//! # Decode-shaped ladders (Layer-2 coverage arc, slice 2)
//!
//! Alongside the square/general ladder, [`Judge::size_plan`] appends
//! **decode-representative** cells on the decode-attention path so the
//! Judge measures the ops at the SKINNY shapes autoregressive decode
//! (seq_q=1) actually runs:
//!
//! - **MatMul** — a `[1,H]×[H,H]` hidden/attention projection GEMV
//!   (bandwidth-bound), distinct in cost regime from the square
//!   (compute-bound) cells.
//! - **MaxReduce / SumReduce** — the row-max and denominator-sum of the
//!   decomposed softmax, reducing a wide `[Hq, k_len]` score row.
//! - **Sub / Exp / Div** — the elementwise softmax components over the
//!   `[Hq, k_len]` score tensor.
//!
//! These feed slice-4's flash-vs-decomposed ranking: arm-0 (the
//! decomposed attention region) is costed from the MEASURED decode
//! latencies of its primitives, so the CUDA flash arm only wins on a
//! real comparison.
//!
//! ## Decode ladder — SizeClass aspect key (reconciled, v4)
//!
//! For **non-matmul** ops [`SizeClass`] is still `log2(total_elements)`,
//! so decode reduction/elementwise cells are placed in buckets that are
//! FREE within their op family (reduce/elementwise sc12 + sc17) — a
//! collision would let the oracle keep the MIN latency across the shared
//! key and poison the cell.
//!
//! For **matmul** the aspect-blindness that once blocked the non-square
//! decode cells is FIXED (SizeClass v4, 2026-07-04). The Judge keyed a
//! matmul cell on `total_elements = m*n` (output) while the ranker's
//! realize-time lookup ([`fuel_dispatch::ranker::compute_static_costs`])
//! keyed on `shapes[0].elem_count() = m*k` (the LHS input) — so every
//! *non-square* matmul lookup missed the profiled cell (square matmuls
//! agreed by accident, `m*k == m*n`). Both sides now derive the key from
//! the operand shapes through the shared [`SizeClass::matmul`] /
//! [`SizeClass::for_op`] helper — packing `(log2(m·n), log2(m),
//! log2(k))` — so the same shape maps to one identical key on producer
//! and consumer, and a GEMV never collides with a same-output-size
//! square (their `log2(m)` bytes differ). The previously DEFERRED decode
//! matmuls — the FFN-width GEMV (`[1,2048]×[2048,5632]`), the QKᵀ score
//! GEMV, the attention-output GEMV — are un-deferred into `size_plan`
//! now that they key correctly. This bumped `PROFILE_REPORT_VERSION`
//! v3 → v4 (fuel-ir + fuel-core + fuel-dispatch reconciliation).
//!
//! # Equivalence-class discipline
//!
//! The Judge profiles one device per equivalence class. A rig with
//! four identical RTX 4090s runs one CUDA profile and the dispatch
//! table shares the result across all four ordinals.
//!
//! # Why latency *and* precision
//!
//! Two ops with identical latency can produce different answers — a
//! fast backend that drifts 1e-3 rel vs reference is fine for most
//! ML workloads but catastrophic for a correctness-critical path
//! like a cross-device equivalence test. Carrying both axes lets a
//! dispatch table serve both "fastest" and "most accurate"
//! criteria without re-profiling.

pub mod cache;
pub mod oracle;
pub use cache::*;
pub use oracle::ProfileJudgeOracle;

use crate::probe::ProbeReport;
use fuel_ir::probe::{BackendId, DeviceDescriptor};
use fuel_ir::{DType, Result, Shape};
use fuel_correctness_fixtures::{
    validate_against_fixture, CorrectnessDrift, CorrectnessFixture, FixtureFile,
    FIXTURE_FILE_VERSION,
};
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::time::Instant;
use std::sync::{Arc, RwLock};

// Re-export the dispatch types (moved to fuel-core-types so
// `fuel-graph-router`'s Router can consume them without depending
// on `fuel-core`). All existing callers `use fuel_core::judge::*`
// keep working — `ProfileReport::save` / `ProfileReport::load` are
// inherent methods on the moved type.
pub use fuel_ir::dispatch::{
    OpKind, ProfileEntry, ProfileReport, SizeClass, PROFILE_REPORT_VERSION,
};

/// Default filename for the persisted profile report.
pub const PROFILE_REPORT_FILENAME: &str = "judge.json";

/// Op kinds the Judge currently profiles. Each entry must have a
/// matching `size_plan` arm (returning a non-empty ladder) and a
/// matching `build_input_graph` arm. New families land here as they
/// ship; the order here is the order entries appear in the persisted
/// report (deterministic for diff-friendly output).
const PROFILED_OPS: &[OpKind] = &[
    OpKind::MatMul,
    OpKind::AddElementwise,
    // --- elementwise unary fanout ---
    OpKind::NegElementwise,
    OpKind::SqrElementwise,
    OpKind::SqrtElementwise,
    OpKind::ExpElementwise,
    OpKind::LogElementwise,
    OpKind::SinElementwise,
    OpKind::CosElementwise,
    OpKind::TanhElementwise,
    OpKind::SigmoidElementwise,
    OpKind::SiluElementwise,
    OpKind::GeluElementwise,
    OpKind::ReluElementwise,
    OpKind::StepElementwise,
    OpKind::RecipElementwise,
    OpKind::AbsElementwise,
    OpKind::FloorElementwise,
    OpKind::CeilElementwise,
    OpKind::RoundElementwise,
    OpKind::SignElementwise,
    OpKind::ErfElementwise,
    OpKind::GeluErfElementwise,
    OpKind::RsqrtElementwise,
    // --- elementwise binary additions ---
    OpKind::PowElementwise,
    OpKind::RemElementwise,
    // --- elementwise binary fanout ---
    OpKind::SubElementwise,
    OpKind::MulElementwise,
    OpKind::DivElementwise,
    OpKind::MaximumElementwise,
    OpKind::MinimumElementwise,
    // --- reductions along one dim ---
    OpKind::SumReduce,
    OpKind::MaxReduce,
    OpKind::MinReduce,
    OpKind::MeanReduce,
    // --- reduce-to-broadcast-target ---
    OpKind::ReduceSumTo,
    OpKind::ReduceMaxTo,
    // --- scalar / clamp / powi ---
    OpKind::Affine,
    OpKind::ClampElementwise,
    OpKind::PowIElementwise,
    // --- fused decode attention (arm-1 of slice-4's flash-vs-decomposed
    // ranking; the decomposed arm-0 primitives are already covered at
    // decode shapes by the elementwise/reduction/matmul families above) ---
    OpKind::FlashAttn,
];

/// Dtypes the Judge profiles for every op in [`PROFILED_OPS`]. The
/// dispatch key ([`ProfileEntry::dtype`]) always carried this axis;
/// this list is what the *runner* iterates (2026-07-04 dtype slice).
///
/// - **F32** — the correctness/reference baseline and the widest
///   kernel coverage.
/// - **F16 / BF16** — the decode/inference dtypes. Flash-attention,
///   skinny GQA matmuls, and the elementwise glue in a transformer
///   forward all run in half precision, so the ranked dispatch table
///   needs their measured per-dtype/per-backend latencies (the coarse
///   Layer-1 model can't tell an f16 kernel from an f32 one).
///
/// **F64 is intentionally excluded**: it has no inference consumer and
/// its only routing path is the correctness-reference backend, which
/// the ranked table never picks. Adding it would triple-count cold
/// cells for no ranking benefit. A cell whose backend lacks a kernel
/// for one of these dtypes is skipped cleanly, so listing a dtype here
/// is safe even on backends with partial coverage.
const PROFILED_DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16];

// =============================================================
// Decode-shaped size-ladder constants (Layer-2 coverage arc,
// slice 2, 2026-07-04)
// =============================================================
//
// The general ladders are square/large — a 1024×1024×1024 matmul,
// a 2²⁰-element reduction. Decode (seq_q=1 autoregressive) runs
// SKINNY shapes that live in a different cost regime: a
// `[1,K]×[K,N]` GEMV is bandwidth-bound, not compute-bound, and the
// decomposed decode-attention softmax reduces/maps over a `[Hq,
// k_len]` score row. These constants shape the decode-representative
// cells the Judge appends so it measures the ops at the shapes decode
// actually runs (slice-4's flash-vs-decomposed comparison ranks on
// arm-0's MEASURED decode primitives, not extrapolated square costs).
//
// **SizeClass discipline.** For non-matmul ops [`SizeClass`] is
// `log2(total_elements)`, aspect-blind — so each decode reduction/
// elementwise cell below lands in a bucket that is FREE within its op
// family (no collision with an existing ladder cell — the oracle keeps
// the MIN latency across a colliding key, which would poison the cell).
// For matmul the v4 aspect key `matmul(m,n,k)` is derived identically
// on the producer (Judge) and consumer (ranker) sides, so the non-square
// FFN/QKᵀ/attn-output GEMVs are un-deferred (no longer square-consistency
// constrained). See the module-level "Decode ladder — SizeClass aspect
// key" note.

/// TinyLlama-class decode attention head count (Hq). The decomposed
/// decode softmax runs one score row per query head at seq_q=1, so the
/// decode reduction/elementwise cells are shaped `[DECODE_HEADS,
/// k_len]`.
const DECODE_HEADS: usize = 32;

/// Small decode key-length (early decode / short context). Wide enough
/// (≥128) to be a genuine row-reduction, distinct from the cols=64
/// general reduction ladder.
const DECODE_KLEN_SMALL: usize = 128;

/// Capacity-ish decode key-length (long context / near KV-cache cap) —
/// the bandwidth-bound end of the decode reduction regime.
const DECODE_KLEN_CAP: usize = 4096;

/// Hidden dim of the decode hidden/attention projection GEMV
/// (`[1,H]×[H,H]`). With the v4 aspect [`SizeClass`] this cell keys as
/// `matmul(1, H, H)` — distinct from every square cell by the `log2(m)`
/// byte (`m=1` vs `m=n`). (Pre-v4 this constant was pinned to keep the
/// old scalar `m·n == m·k` keys in agreement; the aspect key removes
/// that constraint.)
const DECODE_HIDDEN: usize = 2048;

/// TinyLlama-class attention head dim (D). The decode QKᵀ score GEMV
/// contracts over D (`[1,D]×[D,k_len]`), the attention-output GEMV
/// contracts over k_len back down to D (`[1,k_len]×[k_len,D]`).
const DECODE_HEAD_DIM: usize = 64;

/// TinyLlama-class FFN intermediate width. The decode up-projection is a
/// wide GEMV `[1,H]×[H,FFN]` (H=2048 → FFN=5632). Under the *old* scalar
/// key its output `m·n = 5632` bucketed to sc12 — colliding with the
/// 64³ square (`m·n = 4096` → sc12) — which is exactly why slice 2
/// DEFERRED it. The v4 aspect key `matmul(1, 5632, 2048)` no longer
/// collides (the `log2(m)=0` byte separates it from the square's
/// `log2(m)=6`), so it is un-deferred here.
const DECODE_FFN: usize = 5632;

/// Element count of the `[Hq, k_len]` softmax score tensor at the small
/// decode key-length (`Hq * k_len` = 4096 → sc12, free in the
/// elementwise family whose ladder occupies sc {10,16,20}).
const DECODE_SCORE_SMALL_ELEMS: usize = DECODE_HEADS * DECODE_KLEN_SMALL;

/// Element count of the `[Hq, k_len]` softmax score tensor at the
/// capacity decode key-length (`Hq * k_len` = 131072 → sc17, free).
const DECODE_SCORE_CAP_ELEMS: usize = DECODE_HEADS * DECODE_KLEN_CAP;

/// TinyLlama-class decode KV head count (Hkv). GQA folds the 32 query
/// heads onto 4 KV heads (`Hq / Hkv = 8`). Shapes the fused FlashAttn
/// op's `k`/`v` operands `[b, DECODE_KV_HEADS, k_len, d]` — the arm-1
/// half of slice-4's flash-vs-decomposed comparison (arm-0's decomposed
/// QKᵀ/softmax/PV primitives are already covered by slices 2/2.5 at the
/// same decode shapes).
const DECODE_KV_HEADS: usize = 4;

pub fn default_report_path() -> Option<std::path::PathBuf> {
    crate::probe::default_report_path()
        .and_then(|p| p.parent().map(|parent| parent.join(PROFILE_REPORT_FILENAME)))
}

/// How many measurement iterations per (op, dtype, size, backend)
/// cell. Median of this many runs is recorded.
pub const DEFAULT_ITERATIONS: u32 = 7;

/// How many warmup iterations before the timed run. Discards cold-
/// cache and JIT-compile latency; matters most on GPU backends
/// where the first launch includes kernel-cache population.
pub const DEFAULT_WARMUP: u32 = 3;

/// The runner. Consumes a [`ProbeReport`] (so it knows which devices
/// exist and which equivalence classes to share across), builds the
/// full measurement matrix, and produces a [`ProfileReport`].
pub struct Judge {
    pub iterations: u32,
    pub warmup:     u32,
    /// Optional shrunk size ladder. `None` = use the full default
    /// ladder: three square/general sizes per op (up to 1024×1024
    /// matmul / 2²⁰ elementwise), PLUS decode-shaped cells on the
    /// decode-attention path (seq_q=1 matmul GEMV; wide-`k_len`
    /// MaxReduce/SumReduce; decode-score-row Sub/Exp/Div). Tests supply
    /// a shrunk ladder to stay fast.
    pub size_plan_override: Option<Vec<(OpKind, OpSize)>>,
    /// Optional pre-validated correctness fixtures keyed by
    /// `(op, dtype, size_class)`. When a fixture covers a profiling
    /// cell, the Judge validates every CellRun's output against it
    /// via [`validate_against_fixture`] and assigns each
    /// `ProfileEntry::max_rel_error` from the verdict, bypassing the
    /// inline pairwise-consensus path for that cell. Cells without a
    /// matching fixture fall back to the consensus path (today's
    /// default behavior on multi-backend systems).
    ///
    /// One key may map to multiple fixtures when capture used
    /// different `input_seed`s; the Judge applies the first whose
    /// fixture-shape and the cell's regenerated input agree. If no
    /// fixture matches the cell's input seed (mismatch logged), the
    /// fallback consensus path is used for the cell.
    pub fixtures: Option<HashMap<(OpKind, DType, SizeClass), Vec<CorrectnessFixture>>>,
}

impl Default for Judge {
    fn default() -> Self {
        Self {
            iterations: DEFAULT_ITERATIONS,
            warmup:     DEFAULT_WARMUP,
            size_plan_override: None,
            fixtures: None,
        }
    }
}

impl Judge {
    /// Construct a Judge with the default measurement parameters and
    /// the fixture set loaded from `path`. The path may point at a
    /// single `*.json` file or a directory; in the directory case
    /// every `*.json` under the root is loaded recursively (the
    /// `fuel-correctness-fixtures` capture binary writes
    /// `v1/<dtype>/<op>.json`, so a single root covers an entire
    /// fixture distribution).
    ///
    /// Returns an `io::Error` wrapped as a `fuel-core` `Error` if the
    /// path can't be read or any file fails to parse as a
    /// [`FixtureFile`].
    ///
    /// **Use**: pass a fixture root once at Judge construction time;
    /// every cell the Judge profiles thereafter validates against the
    /// fixture (if one exists for that cell) instead of inline
    /// consensus. Cells with no matching fixture fall back to the
    /// consensus path.
    pub fn with_fixtures_from(path: &Path) -> Result<Self> {
        let fixtures = load_fixtures_recursive(path)?;
        let mut judge = Self::default();
        judge.fixtures = Some(fixtures);
        Ok(judge)
    }

    /// Look up a fixture matching `(op, dtype, size_class)` and the
    /// expected element count of the cell's output. Returns `None`
    /// when no fixture set is configured, when no fixture is keyed
    /// for the cell, when no fixture matches the cell's output
    /// shape, when the fixture's `input_seed` doesn't match the
    /// Judge-derived per-cell seed, or when the fixture's
    /// `input_hash` doesn't match the regenerated input bytes.
    ///
    /// **Triple-agreement contract (2026-06-08)**: a fixture is
    /// accepted iff
    ///
    ///   1. `(op, dtype, size_class)` matches the bucket key, AND
    ///   2. `output_element_count` matches the cell's output shape, AND
    ///   3. `input_seed` matches
    ///      [`fuel_correctness_fixtures::capture::derive_seed`]'s
    ///      output for this cell, AND
    ///   4. `input_hash` matches
    ///      [`fuel_correctness_fixtures::capture::hash_f32_input`]
    ///      applied to the regenerated input via
    ///      [`fuel_correctness_fixtures::capture::deterministic_f32_input`].
    ///
    /// Failure of any check is logged and the cell falls back to
    /// inline pairwise consensus (the caller's
    /// `compute_pairwise_consensus` path runs).
    ///
    /// `input_elem_count` is the number of f32 elements the capture
    /// tool's `deterministic_f32_input` would emit for this cell —
    /// the caller derives it from the live `OpSize` via
    /// [`OpSize::input_elements`]. It is decoupled from
    /// `output_element_count` because binary ops carry `[a, b]`
    /// concatenated (input = 2 × N) while their output is a single
    /// `[n]` buffer.
    fn lookup_fixture(
        &self,
        op: OpKind,
        dtype: DType,
        size_class: SizeClass,
        output_elem_count: usize,
        input_elem_count: usize,
    ) -> Option<&CorrectnessFixture> {
        let map = self.fixtures.as_ref()?;
        let bucket = map.get(&(op, dtype, size_class))?;

        // Derive the per-cell seed once — capture's `derive_seed`
        // is a pure function of (op, dtype, size_class).
        let expected_seed =
            fuel_correctness_fixtures::capture::derive_seed(op, dtype, size_class);

        // Lazily regenerate and hash the input only when a fixture's
        // shape + seed agree (the hash is the expensive check).
        let mut regenerated_hash: Option<u64> = None;
        let mut shape_matches = false;
        for f in bucket {
            if f.output_element_count != output_elem_count {
                continue;
            }
            shape_matches = true;
            if f.input_seed != expected_seed {
                eprintln!(
                    "judge: fixture for ({op:?}, {dtype:?}, size_class={}) seed mismatch \
                     (fixture seed={}, expected={}); falling back to consensus",
                    size_class.0, f.input_seed, expected_seed,
                );
                continue;
            }
            let actual_hash = *regenerated_hash.get_or_insert_with(|| {
                let input =
                    fuel_correctness_fixtures::capture::deterministic_f32_input(
                        op,
                        input_elem_count,
                    );
                fuel_correctness_fixtures::capture::hash_f32_input(&input)
            });
            if f.input_hash != actual_hash {
                eprintln!(
                    "judge: fixture input drifted for ({op:?}, {dtype:?}, \
                     size_class={}, seed={}); fixture hash={}, regenerated hash={}; \
                     falling back to consensus",
                    size_class.0, f.input_seed, f.input_hash, actual_hash,
                );
                continue;
            }
            return Some(f);
        }
        if !shape_matches {
            // The bucket exists but no fixture matches the cell's shape.
            // The capture used a different input convention; skip + fall
            // back to consensus rather than emit a misleading rel_err.
            eprintln!(
                "judge: fixture bucket for ({op:?}, {dtype:?}, size_class={}) has no entry \
                 matching output_element_count={output_elem_count}; falling back to consensus",
                size_class.0,
            );
        }
        None
    }

    /// Representative size classes for the Judge's op matrix. Each
    /// entry is (op, [input element counts]) — the Judge picks the
    /// first entry whose op matches and iterates the sizes.
    fn size_plan(&self, op: OpKind) -> Vec<OpSize> {
        if let Some(over) = &self.size_plan_override {
            return over.iter()
                .filter(|(k, _)| *k == op)
                .map(|(_, s)| *s)
                .collect();
        }
        match op {
            OpKind::MatMul => vec![
                // General square ladder (compute-bound regime).
                OpSize::MatMul { m: 64,  n: 64,  k: 64  },
                OpSize::MatMul { m: 256, n: 256, k: 256 },
                OpSize::MatMul { m: 1024, n: 1024, k: 1024 },
                // ---- Decode GEMVs (seq_q=1, bandwidth-bound). ----
                // With the v4 aspect SizeClass every cell below keys as
                // `matmul(m,n,k)` and the same operand dims yield the
                // same key on the ranker's realize-time lookup, so these
                // are found (no longer square-consistency-constrained).
                //
                // Hidden / attention projection `[1,H]×[H,H]`.
                OpSize::MatMul { m: 1, n: DECODE_HIDDEN, k: DECODE_HIDDEN },
                // FFN up-projection `[1,H]×[H,FFN]` (wide GEMV). Un-
                // deferred from slice 2: under the old scalar key its
                // output `m·n=5632` collided with the 64³ square (both
                // sc12); the aspect key `matmul(1,5632,2048)` separates
                // them by the `log2(m)=0` byte.
                OpSize::MatMul { m: 1, n: DECODE_FFN, k: DECODE_HIDDEN },
                // QKᵀ score GEMV, per query head `[1,D]×[D,k_len]` (the
                // leading Hq batch dim doesn't affect the aspect key —
                // `for_op` reads the trailing two dims). Contracts over
                // the head dim D; output width is k_len.
                OpSize::MatMul { m: 1, n: DECODE_KLEN_SMALL, k: DECODE_HEAD_DIM },
                // Attention-output GEMV, per head `[1,k_len]×[k_len,D]`.
                // Contracts over k_len back down to D — the (n,k)-swapped
                // mirror of QKᵀ, and the aspect key keeps them distinct.
                OpSize::MatMul { m: 1, n: DECODE_HEAD_DIM, k: DECODE_KLEN_SMALL },
            ],
            // Element-wise binary + unary share one ladder. Three sizes
            // span the regime where launch overhead dominates (1 KiB),
            // L2 still fits the working set (256 KiB), and DRAM
            // bandwidth dominates (4 MiB).
            OpKind::AddElementwise
            | OpKind::SubElementwise
            | OpKind::MulElementwise
            | OpKind::DivElementwise
            | OpKind::MaximumElementwise
            | OpKind::MinimumElementwise
            | OpKind::PowElementwise
            | OpKind::RemElementwise
            | OpKind::NegElementwise
            | OpKind::SqrElementwise
            | OpKind::SqrtElementwise
            | OpKind::ExpElementwise
            | OpKind::LogElementwise
            | OpKind::SinElementwise
            | OpKind::CosElementwise
            | OpKind::TanhElementwise
            | OpKind::SigmoidElementwise
            | OpKind::SiluElementwise
            | OpKind::GeluElementwise
            | OpKind::ReluElementwise
            | OpKind::StepElementwise
            | OpKind::RecipElementwise
            | OpKind::AbsElementwise
            | OpKind::FloorElementwise
            | OpKind::CeilElementwise
            | OpKind::RoundElementwise
            | OpKind::SignElementwise
            | OpKind::ErfElementwise
            | OpKind::GeluErfElementwise
            | OpKind::RsqrtElementwise
            | OpKind::Affine
            | OpKind::ClampElementwise
            | OpKind::PowIElementwise => {
                let mut ladder = vec![
                    OpSize::Elementwise(1 << 10),
                    OpSize::Elementwise(1 << 16),
                    OpSize::Elementwise(1 << 20),
                ];
                // Decode-score-row cells for the softmax elementwise
                // components (sub/exp/div over the `[Hq, k_len]` score
                // tensor). Only these three ops appear on the decomposed
                // decode-attention softmax path — the rest of the
                // fanout (sin/cos/tanh/gelu/…) keeps the general ladder.
                // Element counts 4096 (sc12) / 131072 (sc17) are both
                // free in this family's occupied buckets {10,16,20}.
                if is_decode_softmax_elementwise(op) {
                    ladder.push(OpSize::Elementwise(DECODE_SCORE_SMALL_ELEMS));
                    ladder.push(OpSize::Elementwise(DECODE_SCORE_CAP_ELEMS));
                }
                ladder
            }
            // Per-axis reductions: probe last-dim reductions over a
            // `[rows, cols]` shape with cols=64 (typical hidden-dim
            // chunk). Total elements 1 KiB / 64 KiB / 1 MiB to align
            // with the elementwise size ladder for cross-family
            // size_class comparison in the dispatch table.
            OpKind::SumReduce
            | OpKind::MaxReduce
            | OpKind::MinReduce
            | OpKind::MeanReduce => {
                let mut ladder = vec![
                    OpSize::Reduce { rows: 1 << 4,  cols: 64 },
                    OpSize::Reduce { rows: 1 << 10, cols: 64 },
                    OpSize::Reduce { rows: 1 << 14, cols: 64 },
                ];
                // Decode-row cells for the softmax reductions: MaxReduce
                // (row max) + SumReduce (denominator). The reduced dim
                // is the last dim (cols = k_len), so these reduce a WIDE
                // `[Hq, k_len]` row — a different regime from the
                // narrow cols=64 general ladder. Total elements
                // 32*128=4096 (sc12) / 32*4096=131072 (sc17) are both
                // free in this family's occupied buckets {10,16,20}.
                // Min/Mean are NOT on the softmax path — no decode cell.
                if is_decode_softmax_reduction(op) {
                    ladder.push(OpSize::Reduce { rows: DECODE_HEADS, cols: DECODE_KLEN_SMALL });
                    ladder.push(OpSize::Reduce { rows: DECODE_HEADS, cols: DECODE_KLEN_CAP });
                }
                ladder
            }
            // Reduce-to-broadcast-target — reduces a `[rows, cols]`
            // input to a `[1, cols]` output (sum/max along leading
            // dim). The canonical autograd-backward shape; broader
            // patterns (multi-axis, mid-rank insertion) follow once
            // the dispatch tables can carry them.
            OpKind::ReduceSumTo
            | OpKind::ReduceMaxTo => vec![
                OpSize::ReduceTo { rows: 1 << 4,  cols: 64 },
                OpSize::ReduceTo { rows: 1 << 10, cols: 64 },
                OpSize::ReduceTo { rows: 1 << 14, cols: 64 },
            ],
            // Fused decode attention (arm-1 of slice-4's flash-vs-
            // decomposed ranking). Two decode-representative GQA cells at
            // seq_q=1: a short-context and a capacity-ish k_len. Both share
            // TinyLlama geometry (Hq=32, Hkv=4, D=64) and key to DISTINCT
            // `SizeClass::attention` buckets (the k_len byte differs), so
            // the oracle costs the short-context and long-context flash
            // kernel separately. Measurement-bounded: at k_len=4096 the
            // naive CPU SDPA is ~Hq·k_len·D ≈ 8.4M MACs per QKᵀ/PV pass.
            OpKind::FlashAttn => vec![
                OpSize::Attention {
                    b: 1, hq: DECODE_HEADS, hkv: DECODE_KV_HEADS,
                    d: DECODE_HEAD_DIM, k_len: DECODE_KLEN_SMALL,
                },
                OpSize::Attention {
                    b: 1, hq: DECODE_HEADS, hkv: DECODE_KV_HEADS,
                    d: DECODE_HEAD_DIM, k_len: DECODE_KLEN_CAP,
                },
            ],
            // OpKind is `#[non_exhaustive]` — future variants land
            // here until the Judge gets a measurement strategy for
            // them. Empty plan = "Judge skips this op silently."
            _ => Vec::new(),
        }
    }

    /// Run the full profile matrix for every equivalence class in
    /// `probe`. Skips backends that aren't compiled in; logs (stderr)
    /// any backend that's present but hasn't been wired into the
    /// Judge yet (e.g. Vulkan in this revision).
    ///
    /// **Reference-retirement note (2026-06-07)**: the loop order is
    /// inverted relative to the pre-retirement shape — outer over
    /// `(op, size)`, inner over equivalence classes. This lets Judge
    /// collect all backends' outputs at a given cell, compute pairwise
    /// consensus across them, and assign each entry's `max_rel_error`
    /// relative to the consensus cluster (no privileged "reference
    /// output" required). When only one backend is available at a
    /// cell, consensus is trivial and `max_rel_error = 0.0` (no peers
    /// to compare against — the caller treats the lone backend as
    /// authoritative).
    ///
    /// **Per-alternative measurement (2026-06-08)**: for each
    /// equivalence-class representative the Judge now walks
    /// `KernelBindingTable::lookup_alternatives(op_kind, dtypes,
    /// backend)` and emits one `CellRun` per registered alternative.
    /// One alternative is timed via the standard realizer path; its
    /// CellRun is attributed to the sibling the picker ACTUALLY
    /// dispatched (the bridge's post-realize report — Session 3
    /// rider, 2026-06-11 — with first-registered fallback when the
    /// plan had no entry for the measured root). The remaining
    /// alternatives at the same `(op, dtypes, backend)` key are
    /// timed via a direct kernel-pointer call so AOCL/MKL siblings
    /// get distinct latency numbers.
    pub fn run(&self, probe: &ProbeReport) -> ProfileReport {
        let mut entries = Vec::new();
        let classes = probe.equivalence_classes();
        // Stable iteration order over equivalence classes — sort
        // class keys so the profile report is deterministic across
        // runs (HashMap iteration order is not guaranteed across
        // process invocations).
        let mut class_keys: Vec<_> = classes.keys().collect();
        class_keys.sort_by_key(|k| (k.backend as u32, k.device_id, k.vendor_id));

        for &op in PROFILED_OPS {
            for sz in self.size_plan(op) {
                // MatMul keys on shape aspect `(m,n,k)` via the shared
                // `SizeClass::matmul` helper the `fuel-dispatch` ranker
                // also reaches (through `SizeClass::for_op`), so a
                // non-square matmul this producer profiles is found by
                // the consumer at realize time — the same operand dims
                // map to one identical key on both sides. Every other op
                // keys on total element count. (SizeClass v4 producer/
                // consumer reconciliation, 2026-07-04.)
                let size_class = match sz {
                    OpSize::MatMul { m, n, k } => SizeClass::matmul(m, n, k),
                    // FlashAttn keys on the attention aspect `(hq, k_len, d)`
                    // via the shared `SizeClass::attention` helper that
                    // slice-4's bake also reaches (through `SizeClass::for_op`
                    // from the flash node's operand shapes), so a decode
                    // attention cell this producer profiles is found by the
                    // consumer at bake time. Every other op keys on total
                    // element count.
                    OpSize::Attention { hq, k_len, d, .. } => {
                        SizeClass::attention(hq, k_len, d)
                    }
                    _ => SizeClass::from_elem_count(sz.total_elements()),
                };

                // Dtype axis (2026-07-04): measure each (op, size) cell
                // once per profiled dtype. The size_class is dtype-
                // independent (element count, not byte count), so it's
                // computed once outside this loop. Consensus is per
                // (op, dtype, size) — f16 outputs cluster against f16
                // peers, never against the f32 measurement of the same
                // op — because `cell_runs` is rebuilt inside the loop.
                for &dtype in PROFILED_DTYPES {
                    // First pass: per equivalence-class representative,
                    // measure latency + capture the kernel's output for
                    // consensus comparison. The realizer path measures
                    // whichever alternative the picker dispatches; its
                    // CellRun already carries that sibling's
                    // `kernel_source` (the bridge's post-realize report,
                    // filled in by `time_op_capturing` — Session 3 rider).
                    let mut cell_runs: Vec<CellRun> = Vec::with_capacity(class_keys.len());
                    for key in &class_keys {
                        let devs = &classes[key];
                        let rep = devs[0];
                        if let Some(run) = self.measure_on_device_capturing(op, dtype, &sz, rep) {
                            let dispatched_source = run.kernel_source.clone();
                            cell_runs.push(run);

                            // Per-alternative measurement: walk the
                            // REMAINING alternatives at the same
                            // `(op_kind, dtypes, backend)` binding-table
                            // cell and time each via a direct
                            // kernel-pointer call. The dispatched
                            // alternative is already covered by the
                            // realizer measurement above.
                            if let Some(extra_runs) =
                                self.measure_extra_alternatives(op, dtype, &sz, rep, &dispatched_source)
                            {
                                for extra in extra_runs {
                                    cell_runs.push(extra);
                                }
                            }
                        }
                    }

                    // Second pass: pick a correctness verdict for each
                    // CellRun. The fixture fast-path is preferred when a
                    // fixture exists for the cell — that's the
                    // pre-validated multi-backend output captured on a
                    // reference rig, so single-backend systems get the
                    // same correctness signal without needing peers
                    // locally. When no fixture is present, fall back to
                    // pairwise consensus across the CellRuns at this
                    // cell.
                    let expected_elem_count = cell_runs.first()
                        .map(|r| r.output.len())
                        .unwrap_or(0);
                    let input_elem_count = sz.input_elements(op);
                    let fixture = self.lookup_fixture(
                        op,
                        dtype,
                        size_class,
                        expected_elem_count,
                        input_elem_count,
                    );
                    let consensus = if fixture.is_none() {
                        compute_pairwise_consensus(&cell_runs)
                    } else {
                        Vec::new() // unused on the fixture path
                    };

                    // Third pass: emit one ProfileEntry per (run, device).
                    // The first alternative's CellRun replicates across
                    // each equivalence-class device; per-alternative
                    // CellRuns from the binding-table direct path
                    // record on the representative's device_index only
                    // (the cross-device sharing convention applies to
                    // backend-level equivalence, not within-cell kernel
                    // sibling differentiation — siblings are
                    // hardware-agnostic at the binding-table layer).
                    for (i, run) in cell_runs.iter().enumerate() {
                        let rel_err = if let Some(f) = fixture {
                            max_rel_err_vs_fixture(&run.output, f)
                        } else {
                            max_rel_err_vs_consensus(&cell_runs, &consensus, i)
                        };
                        // Find the equivalence class this run belongs to
                        // (its representative's backend matches `run.backend`).
                        let class_devs = classes.iter().find_map(|(k, devs)| {
                            if k.backend == run.backend
                                && devs[0].device_index == run.device_index
                            {
                                Some(devs)
                            } else {
                                None
                            }
                        });
                        if let Some(devs) = class_devs {
                            for d in devs {
                                entries.push(ProfileEntry {
                                    op,
                                    dtype,
                                    size_class,
                                    backend: run.backend,
                                    device_index: d.device_index,
                                    latency_ns: run.latency_ns,
                                    iterations: run.iterations,
                                    max_rel_error: rel_err,
                                    kernel_source: run.kernel_source.clone(),
                                });
                            }
                        }
                    }
                }
            }
        }

        ProfileReport { version: PROFILE_REPORT_VERSION, entries }
    }

    /// Walk binding-table alternatives at `(op_kind, dtypes, backend)`
    /// beyond the one the realizer already timed (the picker's
    /// dispatched sibling, `dispatched_kernel_source`) and run a
    /// direct kernel-pointer measurement on each. Returns `None` when
    /// only one alternative is registered (no extra work) or when the
    /// op family is not yet wired into the direct-call path.
    ///
    /// **Why direct call instead of realizer**: the realizer dispatches
    /// through the picker, which selects ONE alternative per realize.
    /// To measure the siblings the picker didn't choose (e.g. AOCL
    /// when MKL won the rank), we have to bypass the picker entirely
    /// and invoke each specific `BindingEntry::kernel` function
    /// pointer with hand-built inputs.
    ///
    /// **Scope**: v1 supports the subset of [`PROFILED_OPS`] where
    /// input/output Storage + Layout + OpParams can be built without
    /// going through the lazy-tensor graph. Today: matmul, elementwise
    /// unary/binary, reductions, reduce-to, affine/clamp/powi. Other
    /// op families return `None` and only the realizer-measured
    /// dispatched alternative is recorded for the cell.
    fn measure_extra_alternatives(
        &self,
        op: OpKind,
        dtype: DType,
        size: &OpSize,
        device: &DeviceDescriptor,
        dispatched_kernel_source: &str,
    ) -> Option<Vec<CellRun>> {
        // Direct kernel-pointer calls are only meaningful on the CPU
        // backend today — CUDA / Vulkan storage handles are backend-
        // specific and the realizer-internal allocator hierarchy
        // doesn't expose a stand-alone "build CUDA storage from f32
        // slice" entry point at the binding-table layer.
        if device.backend != BackendId::Cpu {
            return None;
        }

        // Dtype axis (2026-07-04): the direct-call input builder
        // (`prepare_direct_call_inputs`) materializes f32 CPU storage
        // only, and `read_output_f32` reads it back as f32. For F16/
        // BF16 cells the realizer-measured primary alternative is the
        // sole entry until the direct-call path grows a typed input
        // builder. This is a coverage gap, not a correctness hazard:
        // AOCL/MKL half-precision siblings (the direct-call path's
        // reason to exist) are feature-gated off in the born-red CPU
        // build anyway, so today no sibling is silently dropped.
        if dtype != DType::F32 {
            return None;
        }

        let alternatives = direct_call_alternatives(op, dispatched_kernel_source);
        if alternatives.is_empty() {
            return None;
        }

        // Build the inputs once — every alternative consumes the same
        // input data, so this is shared across the per-alt timing loop.
        let prepared = match prepare_direct_call_inputs(op, size) {
            Some(p) => p,
            None => return None,
        };

        let mut extra: Vec<CellRun> = Vec::with_capacity(alternatives.len());
        for alt in alternatives {
            if let Some(run) = self.time_alternative_direct(
                op, size, device, &prepared, &alt,
            ) {
                extra.push(run);
            }
        }
        if extra.is_empty() { None } else { Some(extra) }
    }

    /// Direct-call timing path. Bypasses the realizer; calls the
    /// supplied `BindingEntry::kernel` function pointer directly with
    /// the prepared inputs. `iterations` median + `warmup` discard
    /// applied identically to the realizer path.
    fn time_alternative_direct(
        &self,
        op: OpKind,
        size: &OpSize,
        device: &DeviceDescriptor,
        prepared: &PreparedDirectCall,
        alt: &DirectCallAlternative,
    ) -> Option<CellRun> {
        use fuel_dispatch::kernel::OpParams as _; // ensure visible in scope

        // Warmup — discard timings. A backend whose kernel panics
        // mid-warmup is treated as if the alternative didn't exist
        // (skip this CellRun); the primary realizer-measured run
        // still represents the cell.
        let kernel = alt.kernel;
        for _ in 0..self.warmup {
            let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let mut outs = vec![Arc::clone(&prepared.output)];
                kernel(&prepared.inputs, &mut outs, &prepared.layouts, &prepared.op_params)
            }));
            if r.is_err() {
                eprintln!(
                    "judge: skipping {op}@{size:?} on {}:{} kernel_source={:?} — kernel panicked in warmup",
                    device.backend, device.device_index, alt.kernel_source,
                );
                return None;
            }
            // Match `time_op_capturing`'s call-twice tolerance for
            // panic-vs-return-Err: an alternative that returns Err is
            // a real signal (the kernel rejected the inputs) — skip
            // rather than abort.
            if let Ok(Err(_)) = r {
                eprintln!(
                    "judge: skipping {op}@{size:?} on {}:{} kernel_source={:?} — kernel returned Err in warmup",
                    device.backend, device.device_index, alt.kernel_source,
                );
                return None;
            }
        }

        let mut timings_ns = Vec::with_capacity(self.iterations as usize);
        let mut captured_output: Option<Vec<f32>> = None;
        for _ in 0..self.iterations {
            let t0 = Instant::now();
            let res = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let mut outs = vec![Arc::clone(&prepared.output)];
                kernel(&prepared.inputs, &mut outs, &prepared.layouts, &prepared.op_params)
            }));
            let elapsed_ns = t0.elapsed().as_nanos() as u64;
            match res {
                Ok(Ok(())) => {}
                _ => {
                    eprintln!(
                        "judge: skipping {op}@{size:?} on {}:{} kernel_source={:?} — direct call failed mid-run",
                        device.backend, device.device_index, alt.kernel_source,
                    );
                    return None;
                }
            }
            timings_ns.push(elapsed_ns);
            if captured_output.is_none() {
                captured_output = Some(read_output_f32(&prepared.output));
            }
        }
        timings_ns.sort_unstable();
        let median = timings_ns[timings_ns.len() / 2];
        let output = captured_output?;

        Some(CellRun {
            backend: device.backend,
            device_index: device.device_index,
            output,
            latency_ns: median,
            iterations: self.iterations,
            kernel_source: alt.kernel_source.to_string(),
        })
    }

    /// Measure one (op, dtype, size) cell on one representative
    /// device, capturing the kernel's output alongside the timing
    /// data so the caller can compute pairwise consensus across
    /// backends. Returns `None` if the backend isn't compiled in,
    /// its constructor errors out, or the realize itself fails.
    ///
    /// **Reference-retirement note (2026-06-07)**: replaces the
    /// pre-retirement `measure_on_device` which compared against
    /// `tensor.realize_f32_reference()` per-cell. The new flow
    /// defers correctness comparison to [`Self::run`]'s consensus
    /// pass.
    fn measure_on_device_capturing(
        &self,
        op: OpKind,
        dtype: DType,
        size: &OpSize,
        device: &DeviceDescriptor,
    ) -> Option<CellRun> {
        // Native-implementation gate (executor-unification Session 2):
        // the bridge realizer dispatches through the picker, whose
        // off-device fallback (picker-arc step 4b) would silently run
        // a missing GPU op on the CPU — and this cell would record a
        // CPU+transfer latency mislabeled as `device.backend`. The
        // legacy executor skipped such cells (its trait method
        // panicked); preserve that contract by requiring a binding-
        // table alternative at the cell before measuring. CPU is
        // exempt: it's the fallback root (nowhere to fall back to —
        // a genuinely missing CPU kernel surfaces as a realize Err
        // below and the cell is skipped through that path).
        //
        // NB: existence-checked via `lookup_alternatives`, NOT via
        // `primary_kernel_source() == ""` — `kernel_source` is a
        // diagnostic tag that legitimately defaults to `""` (the
        // baracuda CUDA registrations are untagged), so an empty tag
        // does not mean an empty cell.
        if device.backend != BackendId::Cpu
            && !has_binding_alternative(op, dtype, device.backend)
        {
            eprintln!(
                "judge: skipping {op}@{size:?} ({dtype:?}) on {}:{} — no binding-table \
                 alternative registered for this backend/dtype",
                device.backend, device.device_index,
            );
            return None;
        }

        // Walk the factory registry instead of naming each backend.
        // A backend that isn't compiled in simply doesn't appear in
        // the registry; one whose constructor fails is logged and
        // skipped per-device.
        let factory = match crate::factories::factory_for(device.backend) {
            Some(f) => f,
            None => {
                eprintln!(
                    "judge: skipping {}:{} — backend not compiled in",
                    device.backend, device.device_index,
                );
                return None;
            }
        };
        let mut realizer = match factory.try_make_realizer(device.device_index) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "judge: skipping {}:{} — factory failed: {e}",
                    device.backend, device.device_index,
                );
                return None;
            }
        };

        let cell = self.time_op_capturing(op, dtype, size, device, realizer.as_mut());
        drop(realizer);
        cell
    }

    /// Build the op's input graph, realize it for warmup + timed
    /// runs, measure latency, capture the first iteration's output.
    /// Returns the captured output alongside the timing data; the
    /// caller computes correctness via consensus across backends
    /// (see [`Self::run`]).
    ///
    /// All realize calls are wrapped in `catch_unwind` — a kernel
    /// that panics on a corner-case input is logged and skipped, not
    /// fatal to the run. Post-Session-2 the realizer also returns
    /// typed `Err`s (the bridge's no-panics contract); an erroring
    /// backend is skipped through the same per-cell path. This is
    /// what makes the Judge safe to expand across the full op
    /// surface ahead of every backend's coverage.
    fn time_op_capturing(
        &self,
        op: OpKind,
        dtype: DType,
        size: &OpSize,
        device: &DeviceDescriptor,
        realizer: &mut dyn crate::factories::LazyRealizer,
    ) -> Option<CellRun> {
        let tensor = match std::panic::catch_unwind(AssertUnwindSafe(|| {
            build_input_graph(op, dtype, size)
        })) {
            Ok(t) => t,
            Err(_) => {
                eprintln!(
                    "judge: skipping {op}@{size:?} — build_input_graph panicked",
                );
                return None;
            }
        };

        // Warmup — discard timings; stabilize kernel caches, warm
        // up any BLAS internal state, fault-in heap, upload Consts
        // into the realizer's persistent cache. If any warmup call
        // panics or errors (e.g. backend doesn't support this op),
        // skip the entire (op, backend) cell.
        for _ in 0..self.warmup {
            match std::panic::catch_unwind(AssertUnwindSafe(|| {
                realizer.realize_capture_f32(&tensor)
            })) {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => {
                    eprintln!(
                        "judge: skipping {op}@{size:?} on {}:{} — backend realize errored: {e}",
                        device.backend, device.device_index,
                    );
                    return None;
                }
                Err(_) => {
                    eprintln!(
                        "judge: skipping {op}@{size:?} on {}:{} — backend realize panicked",
                        device.backend, device.device_index,
                    );
                    return None;
                }
            }
        }

        // Timed iterations — record each and take the median so
        // one pre-empted scheduler tick doesn't skew the number.
        // Capture the first iteration's output for consensus
        // comparison; subsequent iterations discard the output
        // (correctness is deterministic for a given backend + input).
        let mut timings_ns = Vec::with_capacity(self.iterations as usize);
        let mut captured_output: Option<Vec<f32>> = None;
        for _ in 0..self.iterations {
            let t0 = Instant::now();
            let out = match std::panic::catch_unwind(AssertUnwindSafe(|| {
                realizer.realize_capture_f32(&tensor)
            })) {
                Ok(Ok(v)) => v,
                Ok(Err(e)) => {
                    eprintln!(
                        "judge: skipping {op}@{size:?} on {}:{} — backend realize errored mid-run: {e}",
                        device.backend, device.device_index,
                    );
                    return None;
                }
                Err(_) => {
                    eprintln!(
                        "judge: skipping {op}@{size:?} on {}:{} — backend realize panicked mid-run",
                        device.backend, device.device_index,
                    );
                    return None;
                }
            };
            let elapsed_ns = t0.elapsed().as_nanos() as u64;
            timings_ns.push(elapsed_ns);
            if captured_output.is_none() {
                captured_output = Some(out);
            }
        }
        timings_ns.sort_unstable();
        let median = timings_ns[timings_ns.len() / 2];
        let output = captured_output?;

        // Attribution (executor-unification Session 3 rider): the
        // bridge realizer reports the `kernel_source` of the
        // alternative the picker actually dispatched for the measured
        // root. At multi-sibling cells (portable/AOCL/MKL at one CPU
        // key) the dispatched sibling can differ from the
        // first-registered one, so the realizer's report is the only
        // truthful tag. `None` (no plan entry for the root, or a
        // realizer impl without a picker) falls back to the
        // first-registered tag — exactly the alternative the
        // executor's `compile_node` fallback dispatches in that case.
        let kernel_source = match realizer.last_kernel_source() {
            Some(src) => src.to_string(),
            None => primary_kernel_source(op, dtype, device.backend).to_string(),
        };

        Some(CellRun {
            backend: device.backend,
            device_index: device.device_index,
            output,
            latency_ns: median,
            iterations: self.iterations,
            kernel_source,
        })
    }
}

// =============================================================
// Direct-kernel-call infrastructure for per-alternative timing
// (Phase 1.1 of the backend-extensions refactor, 2026-06-08)
// =============================================================

/// One alternative the direct-call path will time. Holds the
/// kernel function pointer and the `kernel_source` tag the resulting
/// CellRun carries.
struct DirectCallAlternative {
    kernel: fuel_dispatch::kernel::KernelRef,
    kernel_source: &'static str,
}

/// Shared inputs/outputs/layouts/op_params for one (op, size) cell.
/// Built once per cell and reused across every alternative's
/// direct-call measurement so the kernels compare on identical
/// inputs (a prerequisite for cross-alternative consensus to mean
/// anything).
struct PreparedDirectCall {
    inputs:    Vec<Arc<RwLock<fuel_memory::Storage>>>,
    output:    Arc<RwLock<fuel_memory::Storage>>,
    layouts:   Vec<fuel_ir::Layout>,
    op_params: fuel_dispatch::kernel::OpParams,
}

/// Look up the `kernel_source` of the FIRST alternative at the
/// binding-table cell. Post-Session-3-rider this is the FALLBACK
/// attribution only — used when the realizer reports no dispatched
/// sibling ([`crate::factories::LazyRealizer::last_kernel_source`]
/// returned `None`, i.e. the plan had no entry for the measured root
/// and the executor's `compile_node` fallback dispatched the
/// first-registered binding). Returns `""` when no alternative is
/// registered at all.
fn primary_kernel_source(op: OpKind, dtype: DType, backend: BackendId) -> &'static str {
    let dtypes = match canonical_binding_dtypes_for(op, dtype) {
        Some(d) => d,
        None => return "",
    };
    let table = fuel_dispatch::dispatch::global_bindings();
    let alts = table.lookup_alternatives(op, &dtypes, backend);
    alts.first().map(|e| e.kernel_source).unwrap_or("")
}

/// Does the binding table hold ANY alternative at the cell's
/// canonical F32 `(op, dtypes, backend)` key? Existence check for
/// the Judge's native-implementation gate — deliberately distinct
/// from [`primary_kernel_source`], whose `""` return conflates
/// "no alternative" with "first alternative carries the default
/// empty kernel_source tag" (untagged registrations are legal and
/// common — every baracuda CUDA binding is untagged).
///
/// Ops without a canonical dtype mapping return `false` — the gate
/// then skips the (non-CPU) cell rather than risk recording an
/// off-device-fallback measurement under the wrong backend label.
/// Unreachable for today's [`PROFILED_OPS`] (all are mapped).
fn has_binding_alternative(op: OpKind, dtype: DType, backend: BackendId) -> bool {
    let dtypes = match canonical_binding_dtypes_for(op, dtype) {
        Some(d) => d,
        None => return false,
    };
    let table = fuel_dispatch::dispatch::global_bindings();
    !table.lookup_alternatives(op, &dtypes, backend).is_empty()
}

/// Collect every binding-table alternative at the cell EXCEPT the
/// dispatched one (already timed via the realizer). Returns a fresh
/// `Vec` of [`DirectCallAlternative`] — empty when the cell has only
/// one alternative or the op isn't direct-call-eligible yet.
fn direct_call_alternatives(
    op: OpKind,
    dispatched_kernel_source: &str,
) -> Vec<DirectCallAlternative> {
    // Direct-call path is F32-only (see `measure_extra_alternatives`'s
    // dtype gate + `prepare_direct_call_inputs`), so the binding key is
    // always the F32 tuple here.
    let dtypes = match canonical_binding_dtypes_for(op, DType::F32) {
        Some(d) => d,
        None => return Vec::new(),
    };
    let table = fuel_dispatch::dispatch::global_bindings();
    let alts = table.lookup_alternatives(op, &dtypes, BackendId::Cpu);
    if alts.len() < 2 {
        return Vec::new();
    }
    // Skip the alternative the realizer already measured — the FIRST
    // entry whose tag matches `dispatched_kernel_source`, wherever it
    // sits in registration order (the picker is free to dispatch a
    // non-first sibling). If no tag matches (off-cell dispatch — a
    // fallback placement ran the root elsewhere), nothing is skipped:
    // every CPU sibling at the cell is then unmeasured work.
    let mut extras = Vec::with_capacity(alts.len().saturating_sub(1));
    let mut skipped_dispatched = false;
    for alt in alts.iter() {
        if !skipped_dispatched && alt.kernel_source == dispatched_kernel_source {
            skipped_dispatched = true;
            continue;
        }
        extras.push(DirectCallAlternative {
            kernel: alt.kernel,
            kernel_source: alt.kernel_source,
        });
    }
    extras
}

/// Map an `OpKind` to its binding-table dtype list (inputs + outputs)
/// at the profiled `dtype`. Every profiled family is uniform-dtype, so
/// the list is just `dtype` repeated per operand slot. Returns `None`
/// for op families the native-implementation gate / direct-call path
/// don't yet support — caller falls back to first-alternative-only
/// measurement at the cell (or skips the non-CPU cell).
fn canonical_binding_dtypes_for(op: OpKind, dtype: DType) -> Option<Vec<DType>> {
    // Most elementwise ops follow `[input..., output]`. Reductions
    // and reduce-to follow `[input, output]`. MatMul is 3 inputs
    // (no — 2 inputs + 1 output): `[lhs, rhs, out] = [T, T, T]`.
    //
    // Dtype axis (2026-07-04): every profiled family in this map is
    // uniform-dtype (all operands share `dtype`), so substituting the
    // profiled dtype `T` for the former hard-coded F32 gives the right
    // binding-table key for F16/BF16 cells too. The mixed-dtype ops
    // (Cast, QMatMul, comparisons with a U8 mask) are not in
    // PROFILED_OPS and fall through to the `None` arm.
    let t = dtype;
    Some(match op {
        OpKind::MatMul => vec![t, t, t],
        // Binary elementwise: 2 inputs + 1 output.
        OpKind::AddElementwise
        | OpKind::SubElementwise
        | OpKind::MulElementwise
        | OpKind::DivElementwise
        | OpKind::MaximumElementwise
        | OpKind::MinimumElementwise
        | OpKind::PowElementwise
        | OpKind::RemElementwise => vec![t, t, t],
        // Unary elementwise: 1 input + 1 output.
        OpKind::NegElementwise
        | OpKind::SqrElementwise
        | OpKind::SqrtElementwise
        | OpKind::ExpElementwise
        | OpKind::LogElementwise
        | OpKind::SinElementwise
        | OpKind::CosElementwise
        | OpKind::TanhElementwise
        | OpKind::SigmoidElementwise
        | OpKind::SiluElementwise
        | OpKind::GeluElementwise
        | OpKind::ReluElementwise
        | OpKind::StepElementwise
        | OpKind::RecipElementwise
        | OpKind::AbsElementwise
        | OpKind::FloorElementwise
        | OpKind::CeilElementwise
        | OpKind::RoundElementwise
        | OpKind::SignElementwise
        | OpKind::ErfElementwise
        | OpKind::GeluErfElementwise
        | OpKind::RsqrtElementwise
        | OpKind::Affine
        | OpKind::ClampElementwise
        | OpKind::PowIElementwise => vec![t, t],
        // Reductions: 1 input + 1 output.
        OpKind::SumReduce
        | OpKind::MaxReduce
        | OpKind::MinReduce
        | OpKind::MeanReduce
        | OpKind::ReduceSumTo
        | OpKind::ReduceMaxTo => vec![t, t],
        // Fused decode attention, NO-alibi key `[q, k, v, out]` (the
        // Judge builds q/k/v without alibi_slopes). This drives the
        // non-CPU native-implementation gate ([`has_binding_alternative`])
        // so a live CUDA/Vulkan run finds the flash binding and populates
        // the GPU arm-1 rows (CUDA flash is f16/bf16-only, so its f32
        // cell has no binding and is skipped cleanly). The direct-call
        // path stays disabled for FlashAttn — `prepare_direct_call_inputs`
        // has no FlashAttn arm, so `measure_extra_alternatives` returns
        // None regardless; this mapping only powers the existence gate +
        // the first-registered `kernel_source` fallback.
        OpKind::FlashAttn => vec![t, t, t, t],
        // All others: not yet wired into the direct-call path. The
        // realizer-measured first alternative is the only entry for
        // the cell until this list expands.
        _ => return None,
    })
}

/// Build the inputs/outputs/layouts/op_params for direct-call timing
/// of one (op, size) cell. Returns `None` for ops not yet supported.
/// Mirrors the input domains used by `build_input_graph` so the
/// direct-call CellRuns produce numerically-comparable output to the
/// realizer-measured primary alternative (consensus needs aligned
/// inputs across CellRuns).
fn prepare_direct_call_inputs(op: OpKind, size: &OpSize) -> Option<PreparedDirectCall> {
    use fuel_ir::Layout;
    use fuel_cpu_backend::CpuStorageBytes;
    use fuel_dispatch::kernel::OpParams;
    use fuel_memory::{BackendStorage, Storage};

    let make_storage = |bytes: CpuStorageBytes| {
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cpu(bytes),
            DType::F32,
        )))
    };

    match (op, *size) {
        (OpKind::MatMul, OpSize::MatMul { m, n, k }) => {
            let a_data: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 1.3e-3).sin()).collect();
            let b_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 1.7e-3).cos()).collect();
            let lhs = make_storage(CpuStorageBytes::from_slice(&a_data));
            let rhs = make_storage(CpuStorageBytes::from_slice(&b_data));
            let out = make_storage(CpuStorageBytes::from_zero_bytes(
                m * n * std::mem::size_of::<f32>(),
            ));
            let layouts = vec![
                Layout::contiguous(Shape::from_dims(&[m, k])),
                Layout::contiguous(Shape::from_dims(&[k, n])),
                Layout::contiguous(Shape::from_dims(&[m, n])),
            ];
            let op_params = OpParams::Matmul {
                lhs_batch_dims: Vec::new(),
                rhs_batch_dims: Vec::new(),
                m, n, k,
            };
            Some(PreparedDirectCall {
                inputs: vec![lhs, rhs],
                output: out,
                layouts,
                op_params,
            })
        }
        (op, OpSize::Elementwise(n)) if is_binary_elementwise(op) => {
            let (a_data, b_data) = binary_inputs(op, n);
            let lhs = make_storage(CpuStorageBytes::from_slice(&a_data));
            let rhs = make_storage(CpuStorageBytes::from_slice(&b_data));
            let out = make_storage(CpuStorageBytes::from_zero_bytes(
                n * std::mem::size_of::<f32>(),
            ));
            let shape = Shape::from_dims(&[n]);
            let layouts = vec![
                Layout::contiguous(shape.clone()),
                Layout::contiguous(shape.clone()),
                Layout::contiguous(shape),
            ];
            Some(PreparedDirectCall {
                inputs: vec![lhs, rhs],
                output: out,
                layouts,
                op_params: OpParams::None,
            })
        }
        (op, OpSize::Elementwise(n)) if is_unary_elementwise(op) => {
            let raw = unary_input(n);
            let needs_nonzero = matches!(
                op,
                OpKind::SqrtElementwise
                | OpKind::LogElementwise
                | OpKind::RecipElementwise
                | OpKind::RsqrtElementwise,
            );
            let data: Vec<f32> = if needs_nonzero {
                raw.into_iter().map(|x| x + 1.5).collect()
            } else {
                raw
            };
            let inp = make_storage(CpuStorageBytes::from_slice(&data));
            let out = make_storage(CpuStorageBytes::from_zero_bytes(
                n * std::mem::size_of::<f32>(),
            ));
            let shape = Shape::from_dims(&[n]);
            let layouts = vec![
                Layout::contiguous(shape.clone()),
                Layout::contiguous(shape),
            ];
            Some(PreparedDirectCall {
                inputs: vec![inp],
                output: out,
                layouts,
                op_params: OpParams::None,
            })
        }
        (op, OpSize::Elementwise(n)) if is_scalar_op(op) => {
            let data = unary_input(n);
            let inp = make_storage(CpuStorageBytes::from_slice(&data));
            let out = make_storage(CpuStorageBytes::from_zero_bytes(
                n * std::mem::size_of::<f32>(),
            ));
            let shape = Shape::from_dims(&[n]);
            let layouts = vec![
                Layout::contiguous(shape.clone()),
                Layout::contiguous(shape),
            ];
            let op_params = match op {
                OpKind::Affine           => OpParams::Affine { mul: 2.0, add: 0.0 },
                OpKind::ClampElementwise => OpParams::Clamp { min: -0.5, max: 0.5 },
                OpKind::PowIElementwise  => OpParams::PowI { exp: 3 },
                _ => unreachable!(),
            };
            Some(PreparedDirectCall {
                inputs: vec![inp],
                output: out,
                layouts,
                op_params,
            })
        }
        // Reductions + ReduceTo not yet wired into the direct-call
        // path: the OpParams shape needs the input layout, and the
        // output shape is op-dependent. Falls back to first-alt-only
        // measurement until the binding-table key conventions are
        // double-checked end-to-end here.
        _ => None,
    }
}

/// Read an output `Storage`'s F32 bytes back out as a `Vec<f32>` for
/// consensus comparison.
fn read_output_f32(out: &Arc<RwLock<fuel_memory::Storage>>) -> Vec<f32> {
    use fuel_memory::BackendStorage;
    let g = out.read().expect("output storage lock");
    match &g.inner {
        BackendStorage::Cpu(c) => c
            .as_slice::<f32>()
            .expect("cpu storage as f32 slice")
            .to_vec(),
        // Direct-call path is CPU-only today; other backends are
        // guarded by `measure_extra_alternatives` returning None
        // before we reach this point. A non-CPU storage here is a
        // bug, but we'd rather skip the entry than panic in the
        // Judge.
        #[allow(unreachable_patterns)]
        _ => Vec::new(),
    }
}

/// Per-op size descriptor. Exposed (not private) so tests can supply
/// a shrunk `size_plan_override`; not in the persisted profile
/// report shape — the report uses the bucketed [`SizeClass`] instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpSize {
    /// Dense matmul `[m, k] @ [k, n] -> [m, n]`.
    MatMul { m: usize, n: usize, k: usize },
    /// Flat 1-D shape used by unary, binary, scalar/affine, clamp, powi
    /// kernels. Probe builds inputs of shape `[n]`.
    Elementwise(usize),
    /// Reduction along the last dim of a 2-D shape `[rows, cols]`.
    /// Total element count = rows × cols. Used for `SumReduce` /
    /// `MaxReduce` / `MinReduce` / `MeanReduce`.
    Reduce { rows: usize, cols: usize },
    /// Reduce-to-broadcast-target from `[rows, cols]` → `[1, cols]`
    /// (sum-reduce-along-rows). Used for `ReduceSumTo` / `ReduceMaxTo`.
    ReduceTo { rows: usize, cols: usize },
    /// Decode-shaped GQA attention for [`OpKind::FlashAttn`]. seq_q is
    /// fixed at 1 (autoregressive decode — the regime slice-4's
    /// flash-vs-decomposed ranking cares about). Builds
    /// `q = [b, hq, 1, d]`, `k = v = [b, hkv, k_len, d]` (`hq % hkv == 0`,
    /// GQA), attending the full `k_len` prefix. Keyed via
    /// [`SizeClass::attention`]`(hq, k_len, d)`.
    Attention { b: usize, hq: usize, hkv: usize, d: usize, k_len: usize },
}

impl OpSize {
    fn total_elements(&self) -> usize {
        match *self {
            OpSize::MatMul { m, n, k: _ } => m * n,
            OpSize::Elementwise(n) => n,
            OpSize::Reduce { rows, cols } => rows * cols,
            OpSize::ReduceTo { rows, cols } => rows * cols,
            // Output element count of decode attention `[b, hq, 1, d]`.
            // Attention keys via `SizeClass::attention`, not this — this
            // is only the Total-key fallback for a non-attention consumer.
            OpSize::Attention { b, hq, d, .. } => b * hq * d,
        }
    }

    /// Element count of the input buffer the capture tool's
    /// [`fuel_correctness_fixtures::capture::deterministic_f32_input`]
    /// would emit for an op of this size. Used by the Judge's
    /// fixture-lookup path to regenerate-and-hash the input.
    ///
    /// Per-op shape (mirrors capture's `deterministic_f32_input`):
    /// - MatMul: `m*k + k*n` — `a_data` followed by `b_data`.
    /// - Elementwise (binary): `2 * n` — `[a, b]` concatenated.
    /// - Elementwise (unary / scalar / clamp / powi): `n`.
    /// - Reduce / ReduceTo: `rows * cols`.
    ///
    /// `op` is needed to disambiguate binary vs unary on the
    /// `Elementwise(n)` shape.
    fn input_elements(&self, op: OpKind) -> usize {
        match *self {
            OpSize::MatMul { m, n, k } => m * k + k * n,
            OpSize::Elementwise(n) => {
                if is_binary_elementwise(op) { 2 * n } else { n }
            }
            OpSize::Reduce { rows, cols } => rows * cols,
            OpSize::ReduceTo { rows, cols } => rows * cols,
            // q + k + v elements. FlashAttn has no capture-tool fixture
            // (not in the fixture distribution), so this feeds no live
            // fixture-hash path today; kept consistent for completeness.
            OpSize::Attention { b, hq, hkv, d, k_len } => {
                b * hq * 1 * d + 2 * (b * hkv * k_len * d)
            }
        }
    }
}

/// Build a leaf [`LazyTensor`] at `dtype` from f32 source `data`. F16/
/// BF16 down-convert the deterministic f32 domain element-wise
/// (`half::f16/bf16::from_f32`) so the same input distribution is
/// exercised at every precision — the cross-dtype comparison stays
/// apples-to-apples modulo the dtype's own rounding. Panics only on a
/// dtype outside the profiled `{F32, F16, BF16}` set (never reached:
/// [`PROFILED_DTYPES`] is the sole caller of `build_input_graph`).
fn make_leaf(dtype: DType, data: Vec<f32>, shape: Shape) -> crate::lazy::LazyTensor {
    use crate::lazy::LazyTensor;
    let dev = crate::Device::cpu();
    match dtype {
        DType::F32 => LazyTensor::from_f32(data, shape, &dev),
        DType::F16 => LazyTensor::from_f16(
            data.iter().map(|&x| half::f16::from_f32(x)).collect::<Vec<_>>(),
            shape,
            &dev,
        ),
        DType::BF16 => LazyTensor::from_bf16(
            data.iter().map(|&x| half::bf16::from_f32(x)).collect::<Vec<_>>(),
            shape,
            &dev,
        ),
        other => panic!("build_input_graph: unsupported profiled dtype {other:?}"),
    }
}

/// Build a same-graph const sibling of `a` at `dtype` from f32 source
/// `data`. Mirrors [`make_leaf`] but through the `const_*_like`
/// constructors so the second operand shares `a`'s graph.
fn make_const_like(
    a: &crate::lazy::LazyTensor,
    dtype: DType,
    data: Vec<f32>,
    shape: Shape,
) -> crate::lazy::LazyTensor {
    match dtype {
        DType::F32 => a.const_f32_like(data, shape),
        DType::F16 => a.const_f16_like(
            data.iter().map(|&x| half::f16::from_f32(x)).collect::<Vec<_>>(),
            shape,
        ),
        DType::BF16 => a.const_bf16_like(
            data.iter().map(|&x| half::bf16::from_f32(x)).collect::<Vec<_>>(),
            shape,
        ),
        other => panic!("build_input_graph: unsupported profiled dtype {other:?}"),
    }
}

/// Build a 1-node graph for the given (op, dtype, size) that takes
/// constant inputs. The inputs are deterministic (generated in f32 and
/// down-converted to `dtype`) so precision comparisons across backends
/// are meaningful.
fn build_input_graph(op: OpKind, dtype: DType, size: &OpSize) -> crate::lazy::LazyTensor {
    match (op, *size) {
        (OpKind::MatMul, OpSize::MatMul { m, n, k }) => {
            let a_data: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 1.3e-3).sin()).collect();
            let b_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 1.7e-3).cos()).collect();
            let a = make_leaf(dtype, a_data, Shape::from_dims(&[m, k]));
            let b = make_const_like(&a, dtype, b_data, Shape::from_dims(&[k, n]));
            a.matmul(&b).unwrap()
        }
        (op, OpSize::Elementwise(n)) if is_binary_elementwise(op) => {
            let (a_data, b_data) = binary_inputs(op, n);
            let a = make_leaf(dtype, a_data, Shape::from_dims(&[n]));
            let b = make_const_like(&a, dtype, b_data, Shape::from_dims(&[n]));
            apply_binary(op, &a, &b)
        }
        // -------- elementwise unary fanout --------
        //
        // Domain notes:
        // - sqrt/log require strictly positive inputs; inputs are in
        //   `[0.5, 2.5]` after the +1.5 offset.
        // - exp/sigmoid saturate quickly; inputs are bounded
        //   `[-1, 1]` (sin output) so exp stays in `[0.36, 2.72]` and
        //   sigmoid stays away from saturation (more measurable
        //   precision differences across backends).
        // - silu/gelu/relu/tanh/neg/sqr/step/sin/cos all accept any f32.
        // -------- per-axis reductions --------
        //
        // Reduce along the last dim of `[rows, cols]`, producing
        // `[rows]`. Bucketing on total element count means the
        // dispatch table groups `Reduce(rows=1024, cols=64)` and
        // `Elementwise(65536)` in the same size class — fine, the
        // size_class is the per-op axis, not a cross-op identity.
        (op, OpSize::Reduce { rows, cols }) if is_reduction(op) => {
            let n = rows * cols;
            let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.7e-3).sin()).collect();
            let a = make_leaf(dtype, data, Shape::from_dims(&[rows, cols]));
            apply_reduction(op, &a)
        }
        // -------- reduce-to-broadcast-target --------
        //
        // Reduce `[rows, cols]` to `[1, cols]`.
        (op, OpSize::ReduceTo { rows, cols }) if is_reduce_to(op) => {
            let n = rows * cols;
            let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.7e-3).sin()).collect();
            let a = make_leaf(dtype, data, Shape::from_dims(&[rows, cols]));
            let target = Shape::from_dims(&[1, cols]);
            match op {
                OpKind::ReduceSumTo => a.reduce_sum_to(target),
                OpKind::ReduceMaxTo => a.reduce_max_to(target),
                _ => unreachable!(),
            }
        }
        // -------- scalar / clamp / powi (one-input non-unary) --------
        //
        // Affine here is the canonical MulScalar form (mul=2.0); the
        // dispatch key OpKind::Affine covers AddScalar too. Clamp uses
        // bounds [-0.5, 0.5] so roughly half the [-1, 1] sin input
        // gets clipped, exercising both branches of the kernel. PowI
        // uses exp=3 so the kernel iterates rather than degenerating
        // to sqr/identity.
        (op, OpSize::Elementwise(n)) if is_scalar_op(op) => {
            let data: Vec<f32> = unary_input(n);
            let a = make_leaf(dtype, data, Shape::from_dims(&[n]));
            match op {
                OpKind::Affine           => a.mul_scalar(2.0),
                OpKind::ClampElementwise => a.clamp(-0.5, 0.5),
                OpKind::PowIElementwise  => a.powi(3),
                _ => unreachable!(),
            }
        }
        (op, OpSize::Elementwise(n)) if is_unary_elementwise(op) => {
            let raw = unary_input(n);
            // Sqrt/Log require strictly positive inputs; Recip can't be
            // measured against zero (1/0 = inf saturates `max_rel_err`).
            // The +1.5 shift puts unary_input's [-1, 1] range into
            // [0.5, 2.5] — safe for all three.
            let needs_nonzero = matches!(
                op,
                OpKind::SqrtElementwise
                | OpKind::LogElementwise
                | OpKind::RecipElementwise
                | OpKind::RsqrtElementwise,
            );
            let data: Vec<f32> = if needs_nonzero {
                raw.into_iter().map(|x| x + 1.5).collect()
            } else {
                raw
            };
            let a = make_leaf(dtype, data, Shape::from_dims(&[n]));
            apply_unary(op, &a)
        }
        // -------- fused decode attention (FlashAttn) --------
        //
        // Builds the GQA decode operands q=[b,hq,1,d], k=v=[b,hkv,k_len,d]
        // and emits `Op::Fused(FLASH_ATTN, FlashAttn{..})` via the
        // LazyTensor `flash_attn` builder (k_len == the K length axis, so
        // the whole prefix is attended — the concrete-k_len decode).
        // seq_q=1 with `causal=true` is a no-op mask (the single query at
        // the last position attends every key), matching decode. On CPU
        // the realize path dispatches this to the registered fused CPU
        // FlashAttn kernel (naive SDPA, `flash_attn_{f32,f16,bf16}` in
        // fuel-cpu-backend::byte_kernels); f16/bf16 inputs down-convert
        // the deterministic f32 domain, mirroring the other families.
        (OpKind::FlashAttn, OpSize::Attention { b, hq, hkv, d, k_len }) => {
            let scale = 1.0 / (d as f32).sqrt();
            let q_elems = b * hq * 1 * d;
            let kv_elems = b * hkv * k_len * d;
            let q_data: Vec<f32> =
                (0..q_elems).map(|i| ((i as f32) * 1.3e-3).sin()).collect();
            let k_data: Vec<f32> =
                (0..kv_elems).map(|i| ((i as f32) * 1.7e-3).cos()).collect();
            let v_data: Vec<f32> =
                (0..kv_elems).map(|i| ((i as f32) * 0.9e-3).sin() + 1.0).collect();
            let q = make_leaf(dtype, q_data, Shape::from_dims(&[b, hq, 1, d]));
            let k = make_const_like(&q, dtype, k_data, Shape::from_dims(&[b, hkv, k_len, d]));
            let v = make_const_like(&q, dtype, v_data, Shape::from_dims(&[b, hkv, k_len, d]));
            q.flash_attn(&k, &v, None, scale, /*causal=*/ true, None, None, None)
                .expect("judge: flash_attn shape/GQA invariant")
        }
        (op, sz) => panic!("build_input_graph: op {op:?} size {sz:?} not implemented"),
    }
}

/// Whether `op` is a one-input scalar / clamp / powi op the Judge
/// profiles using the `Elementwise(n)` size ladder. These differ
/// from the unary ops only in carrying static parameters on the op
/// itself (scalar coefficient, clamp bounds, integer exponent).
fn is_scalar_op(op: OpKind) -> bool {
    matches!(
        op,
        OpKind::Affine
        | OpKind::ClampElementwise
        | OpKind::PowIElementwise,
    )
}

/// Whether `op` is a reduce-to-broadcast-target op the Judge profiles
/// using the `ReduceTo { rows, cols }` size ladder.
fn is_reduce_to(op: OpKind) -> bool {
    matches!(op, OpKind::ReduceSumTo | OpKind::ReduceMaxTo)
}

/// Elementwise ops on the decomposed decode-attention softmax path
/// (`exp(x - max) / sum`): Sub, Exp, Div. These get decode-score-row
/// [`OpSize::Elementwise`] cells appended to their base ladder so the
/// Judge costs the decomposed softmax at the element counts decode
/// runs (slice-4's arm-0 flash-vs-decomposed comparison needs the
/// decomposed primitives' MEASURED decode latencies). Every other
/// elementwise op is off the decode softmax path and keeps only its
/// general ladder.
fn is_decode_softmax_elementwise(op: OpKind) -> bool {
    matches!(
        op,
        OpKind::SubElementwise | OpKind::ExpElementwise | OpKind::DivElementwise,
    )
}

/// Reductions on the decomposed decode-attention softmax path:
/// MaxReduce (row max) and SumReduce (denominator). Min/Mean don't
/// appear there and keep only their general ladder.
fn is_decode_softmax_reduction(op: OpKind) -> bool {
    matches!(op, OpKind::MaxReduce | OpKind::SumReduce)
}

/// Whether `op` is a per-axis reduction the Judge profiles using the
/// `Reduce { rows, cols }` size ladder.
fn is_reduction(op: OpKind) -> bool {
    matches!(
        op,
        OpKind::SumReduce
        | OpKind::MaxReduce
        | OpKind::MinReduce
        | OpKind::MeanReduce,
    )
}

/// Dispatch one per-axis reduction along the last dim. The reduced
/// dim is removed from the output shape.
fn apply_reduction(op: OpKind, a: &crate::lazy::LazyTensor) -> crate::lazy::LazyTensor {
    // Reduce the last dim. Input is rank-2 `[rows, cols]`; output is
    // rank-1 `[rows]`.
    let last_dim = a.rank() - 1;
    match op {
        OpKind::SumReduce  => a.sum_dim(last_dim).unwrap(),
        OpKind::MaxReduce  => a.max_dim(last_dim).unwrap(),
        OpKind::MinReduce  => a.min_dim(last_dim).unwrap(),
        OpKind::MeanReduce => a.mean_dim(last_dim).unwrap(),
        _ => unreachable!("apply_reduction called on non-reduction OpKind {op:?}"),
    }
}

/// Whether `op` is one of the elementwise binary ops the Judge profiles
/// using the `Elementwise(n)` size ladder.
fn is_binary_elementwise(op: OpKind) -> bool {
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

/// Deterministic two-input data for binary ops. `a` is `sin(i*2.1e-3)`
/// and `b` is `cos(i*1.9e-3)` for most ops; for Div, `b` is shifted
/// away from zero (`+ 1.5`, range `[0.5, 2.5]`) to avoid division by
/// near-zero values that would saturate `max_rel_err` to infinity on
/// any tiny precision difference.
fn binary_inputs(op: OpKind, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a: Vec<f32> = (0..n).map(|i| ((i as f32) * 2.1e-3).sin()).collect();
    let mut b: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.9e-3).cos()).collect();
    if matches!(op, OpKind::DivElementwise) {
        for x in &mut b { *x += 1.5; }
    }
    if matches!(op, OpKind::PowElementwise) {
        // Both inputs must be positive: pow(neg, non-int) = NaN under
        // IEEE-754 and saturates max_rel_err. Shift both to [0.5, 2.5].
        for x in &mut a { *x += 1.5; }
        for x in &mut b { *x += 1.5; }
    }
    if matches!(op, OpKind::RemElementwise) {
        // Divisor must be away from zero (`a / b` blows up otherwise).
        for x in &mut b { *x += 1.5; }
    }
    (a, b)
}

/// Dispatch one elementwise binary op against `(a, b)`.
fn apply_binary(
    op: OpKind,
    a: &crate::lazy::LazyTensor,
    b: &crate::lazy::LazyTensor,
) -> crate::lazy::LazyTensor {
    match op {
        OpKind::AddElementwise     => a.add(b).unwrap(),
        OpKind::SubElementwise     => a.sub(b).unwrap(),
        OpKind::MulElementwise     => a.mul(b).unwrap(),
        OpKind::DivElementwise     => a.div(b).unwrap(),
        OpKind::MaximumElementwise => a.maximum(b).unwrap(),
        OpKind::MinimumElementwise => a.minimum(b).unwrap(),
        // Pow/Rem return Result; expect() in Judge's measurement
        // path is fine — inputs are constructed locally here, so a
        // shape/dtype mismatch is a programming bug in the test
        // harness, not a runtime user input.
        OpKind::PowElementwise     => a.pow(b).expect("judge: pow shape/dtype invariant"),
        OpKind::RemElementwise     => a.rem(b).expect("judge: rem shape/dtype invariant"),
        _ => unreachable!("apply_binary called on non-binary OpKind {op:?}"),
    }
}

/// Whether `op` is one of the elementwise unary ops the Judge profiles
/// using the `Elementwise(n)` size ladder.
fn is_unary_elementwise(op: OpKind) -> bool {
    matches!(
        op,
        OpKind::NegElementwise
        | OpKind::SqrElementwise
        | OpKind::SqrtElementwise
        | OpKind::ExpElementwise
        | OpKind::LogElementwise
        | OpKind::SinElementwise
        | OpKind::CosElementwise
        | OpKind::TanhElementwise
        | OpKind::SigmoidElementwise
        | OpKind::SiluElementwise
        | OpKind::GeluElementwise
        | OpKind::ReluElementwise
        | OpKind::StepElementwise
        | OpKind::RecipElementwise
        | OpKind::AbsElementwise
        | OpKind::FloorElementwise
        | OpKind::CeilElementwise
        | OpKind::RoundElementwise
        | OpKind::SignElementwise
        | OpKind::ErfElementwise
        | OpKind::GeluErfElementwise
        | OpKind::RsqrtElementwise,
    )
}

/// Deterministic unary-op input. `sin(i * 2.1e-3)` produces values in
/// `[-1, 1]`, varied enough that argmax/argmin/relu/step all see a
/// realistic mix of positive and negative values.
fn unary_input(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 2.1e-3).sin()).collect()
}

/// Dispatch one elementwise unary op against `a`.
fn apply_unary(op: OpKind, a: &crate::lazy::LazyTensor) -> crate::lazy::LazyTensor {
    match op {
        OpKind::NegElementwise     => a.neg(),
        OpKind::SqrElementwise     => a.sqr(),
        OpKind::SqrtElementwise    => a.sqrt(),
        OpKind::ExpElementwise     => a.exp(),
        OpKind::LogElementwise     => a.log(),
        OpKind::TanhElementwise    => a.tanh(),
        OpKind::SigmoidElementwise => a.sigmoid(),
        OpKind::SiluElementwise    => a.silu(),
        OpKind::GeluElementwise    => a.gelu(),
        OpKind::ReluElementwise    => a.relu(),
        OpKind::SinElementwise     => a.sin(),
        OpKind::CosElementwise     => a.cos(),
        OpKind::StepElementwise    => a.step(),
        OpKind::RecipElementwise   => a.recip(),
        OpKind::AbsElementwise     => a.abs(),
        OpKind::FloorElementwise   => a.floor(),
        OpKind::CeilElementwise    => a.ceil(),
        OpKind::RoundElementwise   => a.round(),
        OpKind::SignElementwise    => a.sign(),
        OpKind::ErfElementwise     => a.erf(),
        OpKind::GeluErfElementwise => a.gelu_erf(),
        OpKind::RsqrtElementwise   => a.rsqrt(),
        _ => unreachable!("apply_unary called on non-unary OpKind {op:?}"),
    }
}

/// Max element-wise relative error between two same-length f32
/// vectors. Denominator guards against divide-by-zero on mutually-
/// zero pairs. Returns 0.0 when the lengths don't match (caller is
/// responsible for length sanity — this function is in the hot
/// path and avoids panics).
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

// =============================================================
// Pairwise consensus — Reference-retirement replacement (2026-06-07)
// =============================================================
//
// Pre-retirement, Judge compared every backend's output against
// `tensor.realize_f32_reference()` to populate `ProfileEntry::max_rel_error`.
// That privileged the `fuel-reference-backend` crate as THE oracle —
// the architectural decision in v0.2 of `05-backend-contract.md`
// demoted Reference but the code lagged.
//
// Post-retirement, Judge runs every available backend's kernel for
// each (op, dtype, size) cell, clusters the outputs by mutual
// `rel_err < CONSENSUS_EPSILON`, and computes each backend's
// `max_rel_error` against the largest cluster (the "consensus
// group"). A backend in the consensus group reports its max
// within-cluster rel_err (typically near 0); a backend outside
// the consensus reports its rel_err against the cluster — that's
// the outlier signal callers want.

/// Generous epsilon for cross-backend consensus clustering.
/// Different backends produce numerically-different bits for the
/// same logical operation (IEEE rounding rules + accumulation
/// order differ across BLAS impls, GPU shader cores, etc.); the
/// cluster definition has to accommodate this. `1e-3` is loose
/// enough that floating-point accumulation reordering won't push
/// honest implementations out of consensus while still catching
/// kernels that are silently wrong.
///
/// Per-op tightening can land later via per-op
/// `PrecisionGuarantee::max_relative` if a particular op family
/// warrants tighter clustering. The 1e-3 default matches the
/// `assert!(e.max_rel_error < 5e-3)` thresholds used in Judge's
/// own tests below — both numbers reflect "this is the noise
/// floor for honest cross-backend implementations of f32 math."
const CONSENSUS_EPSILON: f32 = 1e-3;

/// One backend's measurement for one (op, dtype, size, kernel_source)
/// cell — kernel output captured alongside latency for downstream
/// consensus comparison.
///
/// **Per-alternative measurement (2026-06-08)**: `kernel_source`
/// distinguishes sibling kernels at one `(op, dtypes, backend)`
/// binding-table key (e.g. AOCL vs MKL vs portable-cpu at
/// `(MatMul, [F32×3], BackendId::Cpu)`). Each alternative produces
/// its own CellRun; the pairwise consensus then sees the full
/// cross-kernel-source population, treating AOCL/MKL/portable as
/// peers alongside CUDA/Vulkan.
#[derive(Debug)]
struct CellRun {
    backend: BackendId,
    device_index: u32,
    output: Vec<f32>,
    latency_ns: u64,
    iterations: u32,
    /// Diagnostic tag identifying the kernel sibling at the
    /// binding-table cell. `""` for cells with a single alternative
    /// or when the kernel-source tag is unknown.
    kernel_source: String,
}

/// Compute the largest mutually-close cluster across `runs`. Returns
/// indices into `runs` (sorted ascending) of the consensus members.
///
/// Algorithm: for each run i, build the set of runs that pairwise
/// agree with i (and with each other). The largest such set is the
/// consensus. Ties broken by lowest first-index — deterministic.
///
/// Edge cases:
/// - Empty `runs`: empty consensus.
/// - Single run: trivial consensus of [0]. The lone backend is
///   authoritative by absence of peers.
/// - Two runs agreeing within `CONSENSUS_EPSILON`: consensus is
///   both; each reports its rel_err against the other (small).
/// - Two runs disagreeing: consensus is [0] (first wins ties by
///   index); the other gets the disagreement as its rel_err.
///   Callers should treat 2-backend cells with disagreement as
///   "human review needed" — neither answer is independently
///   validated.
/// - Three runs with one outlier: consensus is the two that agree;
///   the outlier reports its rel_err against the cluster.
fn compute_pairwise_consensus(runs: &[CellRun]) -> Vec<usize> {
    let n = runs.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    // Pairwise rel_err matrix: agree[i][j] = (max_rel_err < CONSENSUS_EPSILON).
    let mut agree = vec![vec![false; n]; n];
    for i in 0..n {
        agree[i][i] = true;
        for j in (i + 1)..n {
            let close = max_rel_err(&runs[i].output, &runs[j].output) < CONSENSUS_EPSILON;
            agree[i][j] = close;
            agree[j][i] = close;
        }
    }

    // For each i, expand a cluster greedily: start with {i}, add
    // each j ∈ neighbors that pairwise-agrees with all current
    // members. Greedy — but for small N (typical: 2-4 backends),
    // exhaustive search would also be cheap. Greedy is correct
    // when the threshold defines a true equivalence-like relation;
    // for f32 cross-backend agreement that's a reasonable assumption.
    let mut best: Vec<usize> = vec![0];
    for i in 0..n {
        let mut cluster = vec![i];
        for j in 0..n {
            if j == i {
                continue;
            }
            // j joins iff agrees with every existing member.
            if cluster.iter().all(|&k| agree[j][k]) {
                cluster.push(j);
            }
        }
        cluster.sort_unstable();
        // Larger cluster wins; ties broken by lower starting index.
        if cluster.len() > best.len() {
            best = cluster;
        }
    }
    best
}

/// Reinterpret an f32 vector as the little-endian bytes that
/// [`validate_against_fixture`] expects, run validation, and lift
/// the verdict into a single rel_err number for the ProfileEntry.
///
/// - `Ok(())` → `0.0` (output matches fixture within tolerance).
/// - `Err(OutOfTolerance { rel_err, .. })` → that `rel_err`.
/// - `Err(LengthMismatch | NonFinite)` → `f32::INFINITY` — these
///   are categorical failures (wrong shape, or NaN/Inf where finite
///   was expected). The dispatch picker will rank such an
///   alternative behind any finite-rel_err peer; the kernel-source
///   tag on the ProfileEntry tells us which alternative blew up.
fn max_rel_err_vs_fixture(out: &[f32], fixture: &CorrectnessFixture) -> f32 {
    let bytes: Vec<u8> = out.iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    match validate_against_fixture(fixture, &bytes) {
        Ok(()) => 0.0,
        Err(CorrectnessDrift::OutOfTolerance { rel_err, .. }) => rel_err as f32,
        Err(CorrectnessDrift::LengthMismatch { .. }) => f32::INFINITY,
        Err(CorrectnessDrift::NonFinite { .. }) => f32::INFINITY,
    }
}

/// Load every `*.json` file under `root` and flatten the contained
/// [`FixtureFile`] entries into a map keyed by
/// `(op, dtype, size_class)`. `root` may itself be a file (one
/// fixture file loaded) or a directory (recursive walk).
///
/// Errors propagate the first I/O failure or deserialization error
/// encountered. A directory with no `*.json` files yields an empty
/// map (the Judge then behaves as if no fixtures were configured).
///
/// Files whose `FixtureFile.version` does not match the current
/// [`FIXTURE_FILE_VERSION`] are skipped with a stderr warning. The
/// load itself is non-fatal — a single stale file in the fixture
/// distribution shouldn't strand the rest of the run.
fn load_fixtures_recursive(
    root: &Path,
) -> Result<HashMap<(OpKind, DType, SizeClass), Vec<CorrectnessFixture>>> {
    let mut map: HashMap<(OpKind, DType, SizeClass), Vec<CorrectnessFixture>> = HashMap::new();
    let mut visit = |p: &Path| -> Result<()> {
        let raw = std::fs::read_to_string(p)
            .map_err(|e| crate::error::Error::Msg(format!(
                "judge: failed to read fixture file {}: {e}", p.display(),
            )))?;
        let file: FixtureFile = serde_json::from_str(&raw)
            .map_err(|e| crate::error::Error::Msg(format!(
                "judge: failed to parse fixture file {} as FixtureFile: {e}",
                p.display(),
            )))?;
        if file.version != FIXTURE_FILE_VERSION {
            eprintln!(
                "judge: skipping fixture file {} — version {} does not match \
                 current FIXTURE_FILE_VERSION ({}); regenerate via fuel-capture-fixtures",
                p.display(),
                file.version,
                FIXTURE_FILE_VERSION,
            );
            return Ok(());
        }
        for f in file.fixtures {
            let key = (f.op, f.dtype, f.size_class);
            map.entry(key).or_default().push(f);
        }
        Ok(())
    };

    fn walk(
        dir: &Path,
        visit: &mut dyn FnMut(&Path) -> Result<()>,
    ) -> Result<()> {
        let entries = std::fs::read_dir(dir).map_err(|e| {
            crate::error::Error::Msg(format!(
                "judge: failed to read fixture dir {}: {e}", dir.display(),
            ))
        })?;
        for entry in entries {
            let entry = entry.map_err(|e| crate::error::Error::Msg(format!(
                "judge: failed reading dir entry in {}: {e}", dir.display(),
            )))?;
            let p = entry.path();
            if p.is_dir() {
                walk(&p, visit)?;
            } else if p.extension().and_then(|s| s.to_str()) == Some("json") {
                visit(&p)?;
            }
        }
        Ok(())
    }

    if root.is_file() {
        visit(root)?;
    } else if root.is_dir() {
        walk(root, &mut visit)?;
    } else {
        return Err(crate::error::Error::Msg(format!(
            "judge: fixture path {} does not exist", root.display(),
        )));
    }
    Ok(map)
}

/// Compute `runs[self_idx]`'s max relative error against the
/// consensus cluster. When the consensus contains only `self_idx`
/// (single backend or every backend disagreed), returns `0.0` —
/// there's no peer to measure drift against.
fn max_rel_err_vs_consensus(
    runs: &[CellRun],
    consensus: &[usize],
    self_idx: usize,
) -> f32 {
    let peers: Vec<usize> = consensus
        .iter()
        .copied()
        .filter(|&i| i != self_idx)
        .collect();
    if peers.is_empty() {
        return 0.0;
    }
    let self_out = &runs[self_idx].output;
    let mut worst = 0.0_f32;
    for i in peers {
        let e = max_rel_err(self_out, &runs[i].output);
        if e > worst {
            worst = e;
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_class_matches_log2_floor() {
        assert_eq!(SizeClass::from_elem_count(1).0, 0);
        assert_eq!(SizeClass::from_elem_count(2).0, 1);
        assert_eq!(SizeClass::from_elem_count(3).0, 1);
        assert_eq!(SizeClass::from_elem_count(4).0, 2);
        assert_eq!(SizeClass::from_elem_count(1 << 16).0, 16);
        assert_eq!(SizeClass::from_elem_count(65_537).0, 16);
    }

    #[test]
    fn size_class_zero_input_clamps_to_one() {
        assert_eq!(SizeClass::from_elem_count(0).0, 0);
    }

    #[test]
    fn op_kind_display_is_stable() {
        assert_eq!(OpKind::MatMul.to_string(), "matmul");
        assert_eq!(OpKind::AddElementwise.to_string(), "add");
    }

    /// Session 3 rider: the realizer-measured CellRun records the
    /// `kernel_source` the realizer REPORTS — the picker's dispatched
    /// sibling — not the binding table's first-registered alternative.
    /// A realizer with no report (`last_kernel_source() == None`,
    /// e.g. the plan had no entry for the root) falls back to the
    /// first-registered tag, matching the executor's `compile_node`
    /// fallback at exactly that decision point.
    #[test]
    fn cell_run_kernel_source_comes_from_realizer_report() {
        struct StubRealizer {
            src: Option<&'static str>,
        }
        impl crate::factories::LazyRealizer for StubRealizer {
            fn realize_f32(
                &mut self,
                _tensor: &crate::lazy::LazyTensor,
            ) -> Result<Vec<f32>> {
                Ok(vec![0.0; 8])
            }
            fn last_kernel_source(&self) -> Option<&'static str> {
                self.src
            }
        }

        let judge = Judge {
            iterations: 1,
            warmup: 0,
            size_plan_override: None,
            fixtures: None,
        };
        let device = DeviceDescriptor {
            backend: BackendId::Cpu,
            device_index: 0,
            hardware_sku: "test-cpu".into(),
            vendor_id: 0,
            device_id: 0,
            compute_capability: None,
            driver_version: String::new(),
            total_memory_bytes: 0,
            location: fuel_ir::DeviceLocation::Cpu,
        };
        let op = OpKind::AddElementwise;
        let size = OpSize::Elementwise(8);

        // Realizer reports a dispatched sibling → CellRun carries it
        // verbatim, regardless of registration order at the cell.
        let mut reporting = StubRealizer { src: Some("stub-sibling") };
        let run = judge
            .time_op_capturing(op, DType::F32, &size, &device, &mut reporting)
            .expect("stub realize succeeds");
        assert_eq!(run.kernel_source, "stub-sibling");

        // No report → first-registered fallback attribution.
        let mut silent = StubRealizer { src: None };
        let run = judge
            .time_op_capturing(op, DType::F32, &size, &device, &mut silent)
            .expect("stub realize succeeds");
        assert_eq!(
            run.kernel_source,
            primary_kernel_source(op, DType::F32, BackendId::Cpu),
            "no realizer report → first-registered binding-table tag",
        );
    }

    #[test]
    fn judge_profiles_cpu_on_small_matmul() {
        // Post-Reference-retirement (2026-06-07): with Reference
        // gone, the local single-backend (CPU) is alone in its
        // (op, dtype, size) cell. Consensus is trivially `[0]`;
        // `max_rel_error` is `0.0` (no peers to measure drift
        // against). The test verifies the Judge still produces
        // entries with sane timing data when only one backend is
        // available — the single-backend edge case in the
        // consensus algorithm.
        let probe = ProbeReport::probe_all();
        let judge = Judge {
            iterations: 3,
            warmup: 1,
            size_plan_override: Some(vec![
                (OpKind::MatMul, OpSize::MatMul { m: 32, n: 32, k: 32 }),
                (OpKind::AddElementwise, OpSize::Elementwise(1 << 10)),
            ]),
            fixtures: None,
        };
        let report = judge.run(&probe);
        assert_eq!(report.version, PROFILE_REPORT_VERSION);
        assert!(report.entries.iter().any(|e| e.backend == BackendId::Cpu));
        // Post-v2 (per-alternative measurement): one CPU cell may
        // produce multiple entries (one per kernel_source — AOCL,
        // MKL, portable-cpu). When peers exist the rel_err can be
        // non-zero but should stay below the bit-stable floor;
        // when only one alternative is registered rel_err is 0.0
        // (no peers → no drift reference).
        for e in report.entries.iter().filter(|e| e.backend == BackendId::Cpu) {
            assert!(e.max_rel_error < 1e-3,
                "cpu cell rel_err should stay tight ({e:?})");
            assert!(e.latency_ns > 0);
        }
    }

    #[test]
    fn judge_profiles_all_unary_elementwise_ops() {
        // Exercise every elementwise-unary OpKind on cpu + reference at
        // a tiny size. Confirms the size_plan / build_input_graph /
        // apply_unary wiring is complete and that no unary kernel
        // diverges wildly from reference on the cpu backend.
        let probe = ProbeReport::probe_all();
        let unary = [
            OpKind::NegElementwise, OpKind::SqrElementwise, OpKind::SqrtElementwise,
            OpKind::ExpElementwise, OpKind::LogElementwise, OpKind::SinElementwise,
            OpKind::CosElementwise, OpKind::TanhElementwise, OpKind::SigmoidElementwise,
            OpKind::SiluElementwise, OpKind::GeluElementwise, OpKind::ReluElementwise,
            OpKind::StepElementwise, OpKind::RecipElementwise, OpKind::AbsElementwise,
            OpKind::FloorElementwise, OpKind::CeilElementwise,
            OpKind::RoundElementwise, OpKind::SignElementwise,
            OpKind::ErfElementwise,   OpKind::GeluErfElementwise,
            OpKind::RsqrtElementwise,
        ];
        let plan: Vec<_> = unary.iter()
            .map(|&op| (op, OpSize::Elementwise(1 << 8)))
            .collect();
        let judge = Judge {
            iterations: 3, warmup: 1,
            size_plan_override: Some(plan),
            fixtures: None,
        };
        let report = judge.run(&probe);
        for &op in &unary {
            let cpu_entries: Vec<_> = report.entries.iter()
                .filter(|e| e.op == op && e.backend == BackendId::Cpu)
                .collect();
            // Post-v2 (per-alternative measurement): ≥1 — when only the
            // portable CPU kernel is registered the count is 1; with
            // AOCL/MKL feature-gated alternatives it grows. Pick the
            // first entry (the realizer-primary measurement) for the
            // numerical-divergence check.
            assert!(!cpu_entries.is_empty(),
                "expected ≥1 cpu entry for {op}, got 0");
            let e = cpu_entries[0];
            // Elementwise unary ops are bit-stable on cpu vs reference
            // (no accumulation order to worry about). Allow a
            // generous bound so transcendental approximations
            // (tanh/sigmoid/silu/gelu) on different math libs still
            // pass — but a runaway divergence (>1e-3) flags a bug.
            assert!(e.max_rel_error < 1e-3,
                "cpu vs reference disagreement on {op}: rel_err={}",
                e.max_rel_error);
            assert!(e.latency_ns > 0);
        }
    }

    #[test]
    fn judge_profiles_all_binary_elementwise_ops() {
        let probe = ProbeReport::probe_all();
        let binary = [
            OpKind::AddElementwise, OpKind::SubElementwise, OpKind::MulElementwise,
            OpKind::DivElementwise, OpKind::MaximumElementwise, OpKind::MinimumElementwise,
            OpKind::PowElementwise, OpKind::RemElementwise,
        ];
        let plan: Vec<_> = binary.iter()
            .map(|&op| (op, OpSize::Elementwise(1 << 8)))
            .collect();
        let judge = Judge {
            iterations: 3, warmup: 1,
            size_plan_override: Some(plan),
            fixtures: None,
        };
        let report = judge.run(&probe);
        for &op in &binary {
            let cpu_entries: Vec<_> = report.entries.iter()
                .filter(|e| e.op == op && e.backend == BackendId::Cpu)
                .collect();
            assert!(!cpu_entries.is_empty(), "expected >=1 cpu entry for {op}, got 0");
            let e = cpu_entries[0];
            assert!(e.max_rel_error < 1e-3,
                "cpu vs reference disagreement on {op}: rel_err={}",
                e.max_rel_error);
            assert!(e.latency_ns > 0);
        }
    }

    #[test]
    fn judge_profiles_all_reductions() {
        let probe = ProbeReport::probe_all();
        let reduce = [
            OpKind::SumReduce, OpKind::MaxReduce,
            OpKind::MinReduce, OpKind::MeanReduce,
        ];
        let plan: Vec<_> = reduce.iter()
            .map(|&op| (op, OpSize::Reduce { rows: 16, cols: 16 }))
            .collect();
        let judge = Judge {
            iterations: 3, warmup: 1,
            size_plan_override: Some(plan),
            fixtures: None,
        };
        let report = judge.run(&probe);
        for &op in &reduce {
            let cpu_entries: Vec<_> = report.entries.iter()
                .filter(|e| e.op == op && e.backend == BackendId::Cpu)
                .collect();
            assert!(!cpu_entries.is_empty(), "expected >=1 cpu entry for {op}, got 0");
            let e = cpu_entries[0];
            // Sum-style reductions accumulate in different order on
            // backend vs reference; allow up to 5e-3.
            assert!(e.max_rel_error < 5e-3,
                "cpu vs reference disagreement on {op}: rel_err={}",
                e.max_rel_error);
            assert!(e.latency_ns > 0);
        }
    }

    #[test]
    fn judge_profiles_all_reduce_to() {
        let probe = ProbeReport::probe_all();
        let reduce_to = [OpKind::ReduceSumTo, OpKind::ReduceMaxTo];
        let plan: Vec<_> = reduce_to.iter()
            .map(|&op| (op, OpSize::ReduceTo { rows: 16, cols: 16 }))
            .collect();
        let judge = Judge {
            iterations: 3, warmup: 1,
            size_plan_override: Some(plan),
            fixtures: None,
        };
        let report = judge.run(&probe);
        for &op in &reduce_to {
            let cpu_entries: Vec<_> = report.entries.iter()
                .filter(|e| e.op == op && e.backend == BackendId::Cpu)
                .collect();
            assert!(!cpu_entries.is_empty(), "expected >=1 cpu entry for {op}, got 0");
            let e = cpu_entries[0];
            assert!(e.max_rel_error < 5e-3,
                "cpu vs reference disagreement on {op}: rel_err={}",
                e.max_rel_error);
            assert!(e.latency_ns > 0);
        }
    }

    #[test]
    fn judge_profiles_all_scalar_ops() {
        let probe = ProbeReport::probe_all();
        let scalar = [
            OpKind::Affine, OpKind::ClampElementwise, OpKind::PowIElementwise,
        ];
        let plan: Vec<_> = scalar.iter()
            .map(|&op| (op, OpSize::Elementwise(1 << 8)))
            .collect();
        let judge = Judge {
            iterations: 3, warmup: 1,
            size_plan_override: Some(plan),
            fixtures: None,
        };
        let report = judge.run(&probe);
        for &op in &scalar {
            let cpu_entries: Vec<_> = report.entries.iter()
                .filter(|e| e.op == op && e.backend == BackendId::Cpu)
                .collect();
            assert!(!cpu_entries.is_empty(), "expected >=1 cpu entry for {op}, got 0");
            let e = cpu_entries[0];
            assert!(e.max_rel_error < 1e-3,
                "cpu vs reference disagreement on {op}: rel_err={}",
                e.max_rel_error);
            assert!(e.latency_ns > 0);
        }
    }

    /// Decode-shaped-ladder slice (Judge Layer-2 coverage arc,
    /// 2026-07-04, slice 2).
    ///
    /// **Born-red**: before this slice the `size_plan` ladders were all
    /// SQUARE/large — no seq_q=1 matmul, no wide-`k_len` reduction, no
    /// decode-score-row elementwise cell. The decomposed decode-
    /// attention softmax (`exp(x-max)/sum`) and the projection/score
    /// GEMVs run at SKINNY shapes that behave nothing like the square
    /// ladder (bandwidth-bound vs compute-bound). This asserts the real
    /// default `size_plan` now carries the decode cells and that the
    /// decode matmul cell keys to a DISTINCT [`SizeClass`] from every
    /// square cell (slice-4's arm-0 flash-vs-decomposed comparison
    /// needs the decomposed primitives measured at decode shapes, keyed
    /// distinctly so the oracle doesn't collapse them onto a square
    /// cell). Exercises the production `size_plan`, not an override.
    #[test]
    fn size_plan_carries_decode_shaped_cells() {
        let judge = Judge::default(); // no override → real size_plan

        // -- MatMul: a seq_q=1 (m==1) decode GEMV must be present, and
        //    its aspect SizeClass must differ from every square cell's.
        //    Keys are derived via `SizeClass::matmul(m,n,k)` — the SAME
        //    helper the producer (`run`) and the ranker consumer reach
        //    (through `for_op`), so this asserts the real dispatch key. --
        let mm = judge.size_plan(OpKind::MatMul);
        let sc_of = |s: &OpSize| match *s {
            OpSize::MatMul { m, n, k } => SizeClass::matmul(m, n, k),
            _ => unreachable!("size_plan(MatMul) yields only MatMul sizes"),
        };
        let square_scs: Vec<SizeClass> = mm
            .iter()
            .filter(|s| matches!(s, OpSize::MatMul { m, n, .. } if m == n))
            .map(sc_of)
            .collect();
        assert!(
            !square_scs.is_empty(),
            "the general square matmul ladder must be preserved",
        );
        let decode_mm: Vec<&OpSize> = mm
            .iter()
            .filter(|s| matches!(s, OpSize::MatMul { m, .. } if *m == 1))
            .collect();
        assert!(
            !decode_mm.is_empty(),
            "size_plan(MatMul) must include a seq_q=1 decode GEMV cell",
        );
        for d in &decode_mm {
            let sc = sc_of(d);
            assert!(
                !square_scs.contains(&sc),
                "decode matmul cell {d:?} SizeClass {sc:?} collides with a \
                 square cell — arm-0 costing would read the square latency",
            );
        }

        // -- Decode-row reductions (softmax row-max + denominator-sum):
        //    a WIDE reduced dim (cols >= 128 = k_len), distinct from the
        //    cols=64 general ladder. Min/Mean are NOT on the softmax
        //    path and must keep only their general ladder. --
        for op in [OpKind::MaxReduce, OpKind::SumReduce] {
            let plan = judge.size_plan(op);
            assert!(
                plan.iter().any(|s| matches!(s, OpSize::Reduce { cols, .. } if *cols >= 128)),
                "size_plan({op}) must include a decode-row (wide-cols) reduction cell",
            );
        }
        for op in [OpKind::MinReduce, OpKind::MeanReduce] {
            let plan = judge.size_plan(op);
            assert!(
                plan.iter().all(|s| matches!(s, OpSize::Reduce { cols, .. } if *cols == 64)),
                "size_plan({op}) is off the decode softmax path — no decode cell expected",
            );
        }

        // -- Decode-score-row elementwise (softmax sub/exp/div over the
        //    [Hq, k_len] score tensor). These three get decode cells
        //    appended; an off-path elementwise op (e.g. Sin) does not. --
        for op in [
            OpKind::SubElementwise,
            OpKind::ExpElementwise,
            OpKind::DivElementwise,
        ] {
            let plan = judge.size_plan(op);
            assert!(
                plan.len() > 3,
                "size_plan({op}) must append decode cells beyond the 3-size base ladder",
            );
        }
        assert_eq!(
            judge.size_plan(OpKind::SinElementwise).len(),
            3,
            "an off-softmax-path elementwise op must keep only its 3-size base ladder",
        );
    }

    /// Slice-2 born-red companion: a fast CPU run over a square + a
    /// decode matmul cell produces `ProfileEntry`s at TWO distinct
    /// [`SizeClass`] keys, each with a positive latency. This is the
    /// end-to-end proof that the decode GEMV is measurable AND
    /// distinguishable from the square regime through the full run
    /// path (deliverable 3's "distinguishable from the square cell").
    #[test]
    fn judge_measures_decode_matmul_distinct_from_square() {
        let probe = ProbeReport::probe_all();
        let judge = Judge {
            iterations: 3,
            warmup: 1,
            size_plan_override: Some(vec![
                // Square (compute-bound) — small enough to stay fast.
                (OpKind::MatMul, OpSize::MatMul { m: 256, n: 256, k: 256 }),
                // Decode GEMV (seq_q=1, bandwidth-bound).
                (OpKind::MatMul, OpSize::MatMul { m: 1, n: 2048, k: 2048 }),
            ]),
            fixtures: None,
        };
        let report = judge.run(&probe);
        // v4 aspect keys — the same `matmul(m,n,k)` the producer's
        // `run` stamps on each entry.
        let square_sc = SizeClass::matmul(256, 256, 256);
        let decode_sc = SizeClass::matmul(1, 2048, 2048);
        assert_ne!(
            square_sc, decode_sc,
            "decode + square matmul must key to different SizeClass",
        );
        let measured = |sc: SizeClass| {
            report.entries.iter().any(|e| {
                e.op == OpKind::MatMul
                    && e.backend == BackendId::Cpu
                    && e.size_class == sc
                    && e.latency_ns > 0
            })
        };
        assert!(measured(square_sc), "square matmul cell must be measured");
        assert!(
            measured(decode_sc),
            "decode matmul cell must be measured with latency > 0 at its own SizeClass",
        );
    }

    /// SizeClass v4 producer/consumer round-trip (Judge Layer-2 coverage
    /// arc, slice 2.5, 2026-07-04).
    ///
    /// **Born-red**: this is the round-trip the aspect-blind key broke.
    /// The Judge (producer) profiled the FFN-width decode GEMV
    /// `[1,2048]×[2048,5632]` keyed on `m·n = 5632`; the `fuel-dispatch`
    /// ranker (consumer) keyed its realize-time lookup on
    /// `shapes[0].elem_count() = m·k = 2048` — a DIFFERENT bucket — so
    /// the lookup never found the produced cell. This drives the real
    /// producer path (`run`) to emit the cell, then performs the exact
    /// consumer key derivation (`SizeClass::for_op` from the operand
    /// shapes, as `compute_static_costs` does) and asserts the produced
    /// entry is HIT through the actual `ProfileJudgeOracle`. Pre-v4 this
    /// asserted-`is_some` lookup returned `None` (miss).
    #[test]
    fn ranker_lookup_hits_producer_decode_gemv_cell() {
        use fuel_dispatch::ranker::JudgeOracle;

        let probe = ProbeReport::probe_all();
        let judge = Judge {
            iterations: 3,
            warmup: 1,
            // The genuinely non-square FFN-width decode GEMV — the cell
            // slice 2 had to DEFER because its old m·n key collided with
            // the 64³ square.
            size_plan_override: Some(vec![
                (OpKind::MatMul, OpSize::MatMul { m: 1, n: 5632, k: 2048 }),
            ]),
            fixtures: None,
        };
        let report = judge.run(&probe);

        // Consumer-side key: derived from the operand shapes
        // lhs=[m,k]/rhs=[k,n] exactly the way the ranker's
        // `compute_static_costs` does (`SizeClass::for_op`).
        let lhs = Shape::from_dims(&[1, 2048]);
        let rhs = Shape::from_dims(&[2048, 5632]);
        let ranker_key = SizeClass::for_op(OpKind::MatMul, &[lhs, rhs]);

        // Old-bug witness: for this non-square shape the pre-v4 scalar
        // keys the two sides used (consumer m·k vs producer m·n) landed
        // in different buckets — the disagreement this slice fixes.
        assert_ne!(
            SizeClass::from_elem_count(1 * 2048), // old consumer key (m·k)
            SizeClass::from_elem_count(1 * 5632), // old producer key (m·n)
            "sanity: non-square shape → the OLD scalar keys disagreed",
        );

        // The produced entry the ranker must now find.
        let cpu_entry = report
            .entries
            .iter()
            .find(|e| {
                e.op == OpKind::MatMul
                    && e.backend == BackendId::Cpu
                    && e.dtype == DType::F32
            })
            .expect("Judge produced a CPU f32 matmul entry for the GEMV");
        assert_eq!(
            cpu_entry.size_class, ranker_key,
            "producer key must EQUAL the consumer's for_op key for the \
             non-square GEMV (the reconciliation)",
        );

        // And the real consumer oracle resolves it — the round trip.
        let oracle = ProfileJudgeOracle::from_report(&report);
        let hit = oracle.measured_latency_ns(
            OpKind::MatMul,
            DType::F32,
            ranker_key,
            BackendId::Cpu,
            cpu_entry.kernel_source.as_str(),
        );
        assert!(
            hit.is_some(),
            "synthetic ranker lookup at the GEMV shape must HIT the \
             produced cell (pre-v4 this missed)",
        );
        assert!(hit.unwrap() > 0, "measured latency must be positive");
    }

    /// Dtype-axis slice (Judge Layer-2 coverage arc, 2026-07-04).
    ///
    /// **Born-red history**: before this slice the Judge hard-coded
    /// `DType::F32` in every measurement call, so `run()` emitted ONLY
    /// f32 `ProfileEntry`s. This test asserts the measurement matrix
    /// now covers f16 AND bf16 for the profiled families, that each
    /// entry is keyed by the dtype it was measured at, and that the
    /// measured latencies are positive. CPU-only shrunk ladder — no
    /// live GPU needed (CPU carries f16/bf16 kernels for the
    /// elementwise + matmul families via the FKC `[F32,F64,BF16,F16]`
    /// fan-out). A live CUDA/Vulkan profile would ADD the per-dtype
    /// GPU rows the ranker ultimately compares flash-vs-decomposed
    /// against, but the axis mechanics are fully exercised on CPU.
    #[test]
    fn judge_profiles_f16_and_bf16_cells() {
        let probe = ProbeReport::probe_all();
        let judge = Judge {
            iterations: 3,
            warmup: 1,
            size_plan_override: Some(vec![
                (OpKind::AddElementwise, OpSize::Elementwise(1 << 8)),
                (OpKind::MulElementwise, OpSize::Elementwise(1 << 8)),
                (OpKind::MatMul, OpSize::MatMul { m: 16, n: 16, k: 16 }),
            ]),
            fixtures: None,
        };
        let report = judge.run(&probe);

        // The half-precision dtypes must each produce ≥1 CPU entry per
        // profiled op, keyed by that dtype, with a positive latency.
        for dtype in [DType::F16, DType::BF16] {
            for op in [
                OpKind::AddElementwise,
                OpKind::MulElementwise,
                OpKind::MatMul,
            ] {
                let cells: Vec<_> = report
                    .entries
                    .iter()
                    .filter(|e| {
                        e.op == op && e.dtype == dtype && e.backend == BackendId::Cpu
                    })
                    .collect();
                assert!(
                    !cells.is_empty(),
                    "dtype-axis: expected ≥1 CPU {dtype:?} entry for {op}, got 0 \
                     (is the {dtype:?} kernel registered for this op?)",
                );
                for e in &cells {
                    assert_eq!(
                        e.dtype, dtype,
                        "entry must be keyed by the dtype it was measured at",
                    );
                    assert!(
                        e.latency_ns > 0,
                        "{op} {dtype:?} latency must be > 0, got {e:?}",
                    );
                    // Single-backend CPU cell → trivial consensus → the
                    // rel_err is 0.0 (no peers). Half-precision rounding
                    // shows up only against a differing peer, which the
                    // CPU-only born-red doesn't have.
                    assert!(
                        e.max_rel_error.is_finite(),
                        "{op} {dtype:?} rel_err must be finite, got {}",
                        e.max_rel_error,
                    );
                }
            }
        }

        // Regression guard: the f32 rows the pre-slice Judge produced
        // are still present — the dtype loop ADDS f16/bf16, it doesn't
        // displace f32.
        assert!(
            report.entries.iter().any(|e| e.dtype == DType::F32
                && e.op == OpKind::AddElementwise
                && e.backend == BackendId::Cpu),
            "dtype-axis: f32 coverage regressed — expected f32 AddElementwise entry",
        );

        // Cross-check: exactly the three profiled dtypes appear, none
        // else (no f64 or stray dtype leaked into the matrix).
        let mut dtypes: Vec<DType> =
            report.entries.iter().map(|e| e.dtype).collect();
        dtypes.sort_by_key(|d| format!("{d:?}"));
        dtypes.dedup();
        assert_eq!(
            dtypes.len(),
            PROFILED_DTYPES.len(),
            "expected exactly the profiled dtype set {PROFILED_DTYPES:?}, saw {dtypes:?}",
        );
        for d in PROFILED_DTYPES {
            assert!(
                dtypes.contains(d),
                "profiled dtype {d:?} missing from the report",
            );
        }
    }

    /// FlashAttn-as-profiled-op slice (Judge Layer-2 coverage arc,
    /// slice 3, 2026-07-04).
    ///
    /// **Born-red**: before this slice `OpKind::FlashAttn` was absent from
    /// [`PROFILED_OPS`], so `run()`'s outer `for &op in PROFILED_OPS` loop
    /// never processed it — a `size_plan_override` carrying a FlashAttn
    /// cell was filtered out and the report held ZERO FlashAttn entries.
    /// This asserts the fused decode-attention op is now MEASURED on CPU:
    /// the Judge builds q=[b,hq,1,d] / k=v=[b,hkv,k_len,d], emits the
    /// `Op::Fused(FLASH_ATTN)` node, realizes it (the registered fused CPU
    /// SDPA kernel), and produces `ProfileEntry`s keyed by the flash
    /// node's `SizeClass::attention` — arm-1 of slice-4's flash-vs-
    /// decomposed comparison, whose latency is what THIS slice measures.
    ///
    /// Two decode GQA cells (short + capacity k_len) must each land at a
    /// DISTINCT attention SizeClass with a positive latency; the same
    /// `attention(hq, k_len, d)` key slice-4's bake derives via `for_op`.
    /// CPU-only born-red; f32 is asserted (the reference-width kernel),
    /// f16/bf16 are checked opportunistically (the half kernels exist,
    /// but a missing dtype cell is skipped cleanly, never fatal).
    #[test]
    fn judge_profiles_flash_attn_at_decode_shapes() {
        let probe = ProbeReport::probe_all();
        // TinyLlama-ish decode GQA: Hq=32, Hkv=4, D=64. Two k_len cells
        // (short + capacity) — kept small enough to stay fast.
        let small = OpSize::Attention { b: 1, hq: 32, hkv: 4, d: 64, k_len: 128 };
        let cap   = OpSize::Attention { b: 1, hq: 32, hkv: 4, d: 64, k_len: 512 };
        let judge = Judge {
            iterations: 3,
            warmup: 1,
            size_plan_override: Some(vec![
                (OpKind::FlashAttn, small),
                (OpKind::FlashAttn, cap),
            ]),
            fixtures: None,
        };
        let report = judge.run(&probe);

        // The two decode cells key to DISTINCT attention SizeClasses —
        // the same helper the producer stamps and slice-4's `for_op`
        // consumer derives.
        let sc_small = SizeClass::attention(32, 128, 64);
        let sc_cap   = SizeClass::attention(32, 512, 64);
        assert_ne!(
            sc_small, sc_cap,
            "short + capacity decode attention must key to different SizeClass",
        );

        // f32 (reference-width) MUST be measured at BOTH decode cells.
        for (label, sc) in [("short", sc_small), ("capacity", sc_cap)] {
            let hit = report.entries.iter().any(|e| {
                e.op == OpKind::FlashAttn
                    && e.backend == BackendId::Cpu
                    && e.dtype == DType::F32
                    && e.size_class == sc
                    && e.latency_ns > 0
            });
            assert!(
                hit,
                "FlashAttn f32 {label} decode cell (SizeClass {sc:?}) must be \
                 measured with latency > 0 — arm-1 of the flash-vs-decomposed \
                 comparison",
            );
        }

        // Consumer round-trip: the key slice-4's bake derives from the
        // flash node's operand shapes via `for_op` must EQUAL the
        // producer's stamped key, so the bake's lookup HITS the cell.
        let consumer_key = SizeClass::for_op(
            OpKind::FlashAttn,
            &[
                Shape::from_dims(&[1, 32, 1, 64]),
                Shape::from_dims(&[1, 4, 128, 64]),
                Shape::from_dims(&[1, 4, 128, 64]),
            ],
        );
        assert_eq!(
            consumer_key, sc_small,
            "the flash node's for_op key must equal the producer's attention key",
        );

        // Half-precision cells exist opportunistically (the CPU flash
        // kernel carries f16/bf16 variants). Not fatal if a backend
        // skips one, but on this CPU build they should all measure.
        for dtype in [DType::F16, DType::BF16] {
            let hit = report.entries.iter().any(|e| {
                e.op == OpKind::FlashAttn
                    && e.backend == BackendId::Cpu
                    && e.dtype == dtype
                    && e.size_class == sc_small
                    && e.latency_ns > 0
            });
            assert!(
                hit,
                "FlashAttn {dtype:?} short decode cell must be measured (the CPU \
                 flash kernel has an {dtype:?} variant)",
            );
        }
    }

    #[test]
    fn dispatch_table_built_from_expanded_report_serves_multiple_kinds() {
        // Confirms the DispatchTable's O(1) lookup path handles the
        // expanded OpKind coverage. The route picker can now pick
        // among many more (op, size_class) cells than the original
        // matmul + add report supported.
        use fuel_ir::dispatch::{Criterion, DispatchTable};

        let probe = ProbeReport::probe_all();
        let judge = Judge {
            iterations: 3, warmup: 1,
            size_plan_override: Some(vec![
                (OpKind::MatMul, OpSize::MatMul { m: 32, n: 32, k: 32 }),
                (OpKind::AddElementwise, OpSize::Elementwise(1 << 8)),
                (OpKind::ReluElementwise, OpSize::Elementwise(1 << 8)),
                (OpKind::SumReduce, OpSize::Reduce { rows: 16, cols: 16 }),
                (OpKind::ReduceSumTo, OpSize::ReduceTo { rows: 16, cols: 16 }),
                (OpKind::Affine, OpSize::Elementwise(1 << 8)),
            ]),
            fixtures: None,
        };
        let report = judge.run(&probe);
        let table = DispatchTable::build(&report);

        // Every kind in the plan should yield a Pick at every
        // criterion. SizeClass(8) covers the elementwise plan
        // (256 elems = 2^8) and the reductions (16*16 = 2^8); the
        // matmul cell keys on its v4 aspect key `matmul(32,32,32)`
        // (the producer no longer keys it on the scalar 32*32=2^10).
        let elementwise_kinds = [
            OpKind::AddElementwise, OpKind::ReluElementwise,
            OpKind::SumReduce, OpKind::ReduceSumTo, OpKind::Affine,
        ];
        for &op in &elementwise_kinds {
            for &crit in &[Criterion::Fastest, Criterion::MostAccurate, Criterion::Balanced] {
                let pick = table.pick(op, DType::F32, SizeClass(8), crit);
                assert!(pick.is_some(),
                    "DispatchTable missing pick for {op}@2^8 / {crit:?}");
            }
        }
        let matmul_pick = table.pick(
            OpKind::MatMul, DType::F32, SizeClass::matmul(32, 32, 32), Criterion::Fastest,
        );
        assert!(matmul_pick.is_some(), "DispatchTable missing matmul pick");

        // The table's keys() should include every (op, dtype,
        // size_class) tuple exercised by the plan.
        let keys = table.keys();
        assert!(keys.iter().any(|(op, _, sc)| *op == OpKind::ReluElementwise && sc.0 == 8));
        assert!(keys.iter().any(|(op, _, sc)| *op == OpKind::SumReduce && sc.0 == 8));
        assert!(keys.iter().any(|(op, _, sc)| *op == OpKind::ReduceSumTo && sc.0 == 8));
        assert!(keys.iter().any(|(op, _, sc)| *op == OpKind::Affine && sc.0 == 8));
    }

    #[test]
    fn profile_report_save_load_roundtrip() {
        let report = ProfileReport {
            version: PROFILE_REPORT_VERSION,
            entries: vec![ProfileEntry {
                op: OpKind::MatMul,
                dtype: DType::F32,
                size_class: SizeClass(8),
                backend: BackendId::Cpu,
                device_index: 0,
                latency_ns: 12_345,
                iterations: 7,
                max_rel_error: 1e-7,
                kernel_source: String::new(),
            }],
        };
        let tmp = std::env::temp_dir().join(format!(
            "fuel-judge-test-{}.json", std::process::id()
        ));
        report.save(&tmp).expect("save");
        let loaded = ProfileReport::load(&tmp).expect("load").expect("file exists");
        assert_eq!(loaded, report);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn max_rel_err_matches_known_values() {
        assert_eq!(max_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        // f32 rounding in both `2.001` literals and the subtraction
        // means the exact numeric answer is not `0.001 / 2.001`. Pin
        // the answer loosely: should be within a few ULP of that
        // ratio and close to 5e-4.
        let got = max_rel_err(&[1.0, 2.0], &[1.0, 2.001]);
        assert!(
            (got - 5.0e-4).abs() < 1e-6,
            "expected ~0.0005, got {got}",
        );
        // Length mismatch → infinity.
        assert!(max_rel_err(&[1.0], &[1.0, 1.0]).is_infinite());
        // NaN in either operand → infinity.
        assert!(max_rel_err(&[f32::NAN], &[0.0]).is_infinite());
    }

    /// Fixture-fast-path integration: when a fixture exists for the
    /// cell, the Judge derives `max_rel_error` from
    /// `validate_against_fixture` instead of running inline pairwise
    /// consensus. This test constructs an intentionally-divergent
    /// fixture for the AddElementwise cell and asserts the divergence
    /// shows up as the entries' `max_rel_error`.
    ///
    /// We deliberately *don't* use a fixture matching the actual cpu
    /// kernel's output here — the assertion is on the wire-up, not on
    /// any specific numerical value the cpu produces. Real fixtures
    /// land via the capture pipeline; this test verifies the path.
    #[test]
    fn judge_uses_fixture_when_provided() {
        use fuel_correctness_fixtures::{
            CorrectnessFixture, ToleranceBand,
        };

        let probe = ProbeReport::probe_all();
        let elem_count = 1 << 10;
        let sc = SizeClass::from_elem_count(elem_count);
        // Plant an obviously-wrong fixture: all zeros expected. The
        // cpu add kernel produces sin+cos = non-zero, so
        // validate_against_fixture's OutOfTolerance.rel_err lands in
        // every entry's max_rel_error (well above any plausible
        // cross-backend drift floor).
        //
        // The fixture must satisfy the triple-agreement contract
        // (seed + hash + shape) for `lookup_fixture` to return it.
        // We use the same `derive_seed` / `deterministic_f32_input` /
        // `hash_f32_input` the capture tool uses, so the lookup path
        // accepts the (deliberately-wrong-expected-output) fixture.
        let zeros: Vec<u8> = vec![0u8; elem_count * 4];
        let seed = fuel_correctness_fixtures::capture::derive_seed(
            OpKind::AddElementwise, DType::F32, sc,
        );
        // AddElementwise is binary: input is the concatenated `[a, b]`
        // buffer of length `2 * n`.
        let input = fuel_correctness_fixtures::capture::deterministic_f32_input(
            OpKind::AddElementwise, 2 * elem_count,
        );
        let input_hash =
            fuel_correctness_fixtures::capture::hash_f32_input(&input);
        let bogus_fixture = CorrectnessFixture {
            op: OpKind::AddElementwise,
            dtype: DType::F32,
            size_class: sc,
            input_seed: seed,
            input_hash,
            expected_output: zeros,
            output_element_count: elem_count,
            // Tight tolerance — guarantees the all-zeros fixture
            // disagrees with any honest add(sin, cos) output.
            tolerance: ToleranceBand {
                max_relative: 1e-9,
                max_absolute: 1e-12,
            },
        };
        let mut map: HashMap<(OpKind, DType, SizeClass), Vec<CorrectnessFixture>> =
            HashMap::new();
        map.insert(
            (OpKind::AddElementwise, DType::F32, SizeClass::from_elem_count(elem_count)),
            vec![bogus_fixture],
        );

        let judge = Judge {
            iterations: 3,
            warmup: 1,
            size_plan_override: Some(vec![
                (OpKind::AddElementwise, OpSize::Elementwise(elem_count)),
            ]),
            fixtures: Some(map),
        };
        let report = judge.run(&probe);
        // Dtype-axis note (2026-07-04): the planted fixture is keyed
        // `(AddElementwise, F32, sc)`, so only the F32 cells hit the
        // fixture fast-path. The f16/bf16 cells the dtype loop now also
        // emits have NO fixture → they take the consensus path → lone
        // CPU backend → rel_err 0.0. Scope this assertion to the F32
        // entries the fixture actually governs.
        let cpu_entries: Vec<_> = report.entries.iter()
            .filter(|e| e.op == OpKind::AddElementwise
                && e.dtype == DType::F32
                && e.backend == BackendId::Cpu)
            .collect();
        assert!(!cpu_entries.is_empty(),
            "expected ≥1 cpu f32 entry for AddElementwise, got 0");
        // The bogus all-zeros fixture is wildly wrong vs the honest
        // sin+cos output, so every cpu entry's max_rel_error should
        // be very large (≥ 1.0 — relative error vs ~zero expected is
        // unbounded; floor at 1.0 is conservative). Without the
        // fixture fast-path this value would be ~0.0 (lone CPU
        // backend → trivial consensus → no peers → 0.0).
        for e in &cpu_entries {
            assert!(e.max_rel_error >= 1.0,
                "expected fixture-derived rel_err ≥1.0 (vs all-zeros fixture), got {} for entry {e:?}",
                e.max_rel_error);
        }
    }

    /// Loader: `with_fixtures_from(file)` reads a JSON fixture file
    /// and produces a Judge whose `fixtures` map has the entry keyed
    /// correctly.
    #[test]
    fn judge_with_fixtures_from_loads_json_file() {
        use fuel_correctness_fixtures::{
            CorrectnessFixture, FixtureFile, ToleranceBand,
            FIXTURE_FILE_VERSION,
        };

        let fixture = CorrectnessFixture {
            op: OpKind::MatMul,
            dtype: DType::F32,
            size_class: SizeClass(10),
            input_seed: 42,
            input_hash: 0xdeadbeef,
            expected_output: vec![0u8; 16],
            output_element_count: 4,
            tolerance: ToleranceBand::F32_DEFAULT,
        };
        let file = FixtureFile {
            version: FIXTURE_FILE_VERSION,
            fixtures: vec![fixture],
        };
        let dir = std::env::temp_dir().join(format!(
            "fuel-judge-fixture-load-{}", std::process::id(),
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("matmul_f32.json");
        std::fs::write(&path, serde_json::to_string(&file).unwrap()).unwrap();

        let judge = Judge::with_fixtures_from(&dir).expect("load");
        let map = judge.fixtures.as_ref().expect("fixtures populated");
        let key = (OpKind::MatMul, DType::F32, SizeClass(10));
        let bucket = map.get(&key).expect("matmul fixture in bucket");
        assert_eq!(bucket.len(), 1);
        assert_eq!(bucket[0].input_seed, 42);

        // Loading from a single file path works too.
        let judge2 = Judge::with_fixtures_from(&path).expect("load file");
        let map2 = judge2.fixtures.as_ref().expect("fixtures populated");
        assert!(map2.contains_key(&key));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Loader: fixture files whose `version` doesn't match the
    /// current `FIXTURE_FILE_VERSION` are skipped with a stderr
    /// warning. The load itself is non-fatal — a single stale file
    /// in the distribution shouldn't strand the rest of the run.
    #[test]
    fn judge_load_skips_mismatched_version() {
        use fuel_correctness_fixtures::{
            CorrectnessFixture, FixtureFile, ToleranceBand,
        };

        let good_fixture = CorrectnessFixture {
            op: OpKind::MatMul,
            dtype: DType::F32,
            size_class: SizeClass(10),
            input_seed: 1,
            input_hash: 2,
            expected_output: vec![0u8; 16],
            output_element_count: 4,
            tolerance: ToleranceBand::F32_DEFAULT,
        };
        let bad_version_file = FixtureFile {
            version: FIXTURE_FILE_VERSION + 999,
            fixtures: vec![good_fixture.clone()],
        };
        let good_version_file = FixtureFile {
            version: FIXTURE_FILE_VERSION,
            fixtures: vec![CorrectnessFixture {
                op: OpKind::AddElementwise,
                ..good_fixture
            }],
        };
        let dir = std::env::temp_dir().join(format!(
            "fuel-judge-fixture-version-{}", std::process::id(),
        ));
        let _ = std::fs::create_dir_all(&dir);
        let bad_path = dir.join("matmul_f32.json");
        let good_path = dir.join("add_f32.json");
        std::fs::write(&bad_path, serde_json::to_string(&bad_version_file).unwrap())
            .unwrap();
        std::fs::write(&good_path, serde_json::to_string(&good_version_file).unwrap())
            .unwrap();

        let judge = Judge::with_fixtures_from(&dir).expect("load");
        let map = judge.fixtures.as_ref().expect("fixtures populated");
        // The bad-version file is silently dropped; the good-version
        // file's bucket is still present.
        assert!(
            !map.contains_key(&(OpKind::MatMul, DType::F32, SizeClass(10))),
            "expected mismatched-version file to be skipped",
        );
        assert!(
            map.contains_key(&(OpKind::AddElementwise, DType::F32, SizeClass(10))),
            "expected good-version file to be loaded",
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `lookup_fixture` returns `None` for a fixture whose `input_seed`
    /// doesn't match the Judge-derived per-cell seed. The caller (run)
    /// then falls back to inline consensus.
    #[test]
    fn lookup_fixture_skips_seed_mismatch() {
        use fuel_correctness_fixtures::{CorrectnessFixture, ToleranceBand};

        let op = OpKind::AddElementwise;
        let dtype = DType::F32;
        let elem_count = 1 << 10;
        let sc = SizeClass::from_elem_count(elem_count);
        let input = fuel_correctness_fixtures::capture::deterministic_f32_input(
            op, 2 * elem_count,
        );
        let input_hash = fuel_correctness_fixtures::capture::hash_f32_input(&input);
        // Plant the right hash but the wrong seed — the lookup must
        // skip the fixture even though the hash would have matched.
        let real_seed = fuel_correctness_fixtures::capture::derive_seed(op, dtype, sc);
        let wrong_seed = real_seed.wrapping_add(1);
        let fixture = CorrectnessFixture {
            op,
            dtype,
            size_class: sc,
            input_seed: wrong_seed,
            input_hash,
            expected_output: vec![0u8; elem_count * 4],
            output_element_count: elem_count,
            tolerance: ToleranceBand::F32_DEFAULT,
        };
        let mut map: HashMap<(OpKind, DType, SizeClass), Vec<CorrectnessFixture>> =
            HashMap::new();
        map.insert((op, dtype, sc), vec![fixture]);
        let judge = Judge { fixtures: Some(map), ..Judge::default() };
        let got = judge.lookup_fixture(op, dtype, sc, elem_count, 2 * elem_count);
        assert!(
            got.is_none(),
            "fixture with mismatched seed should be skipped (got {got:?})",
        );
    }

    /// `lookup_fixture` returns `None` for a fixture whose `input_hash`
    /// doesn't match the regenerated input. The caller falls back to
    /// inline consensus.
    #[test]
    fn lookup_fixture_skips_hash_mismatch() {
        use fuel_correctness_fixtures::{CorrectnessFixture, ToleranceBand};

        let op = OpKind::AddElementwise;
        let dtype = DType::F32;
        let elem_count = 1 << 10;
        let sc = SizeClass::from_elem_count(elem_count);
        let real_seed =
            fuel_correctness_fixtures::capture::derive_seed(op, dtype, sc);
        let input = fuel_correctness_fixtures::capture::deterministic_f32_input(
            op, 2 * elem_count,
        );
        let real_hash =
            fuel_correctness_fixtures::capture::hash_f32_input(&input);
        // Plant the right seed but a wrong hash. The lookup must skip.
        let wrong_hash = real_hash.wrapping_add(0xfeed_face);
        let fixture = CorrectnessFixture {
            op,
            dtype,
            size_class: sc,
            input_seed: real_seed,
            input_hash: wrong_hash,
            expected_output: vec![0u8; elem_count * 4],
            output_element_count: elem_count,
            tolerance: ToleranceBand::F32_DEFAULT,
        };
        let mut map: HashMap<(OpKind, DType, SizeClass), Vec<CorrectnessFixture>> =
            HashMap::new();
        map.insert((op, dtype, sc), vec![fixture]);
        let judge = Judge { fixtures: Some(map), ..Judge::default() };
        let got = judge.lookup_fixture(op, dtype, sc, elem_count, 2 * elem_count);
        assert!(
            got.is_none(),
            "fixture with drifted input_hash should be skipped (got {got:?})",
        );
    }

    /// `lookup_fixture` accepts a fixture when (op, dtype, size_class) +
    /// output shape + seed + input_hash all agree. Triple-agreement
    /// contract.
    #[test]
    fn lookup_fixture_accepts_triple_agreement() {
        use fuel_correctness_fixtures::{CorrectnessFixture, ToleranceBand};

        let op = OpKind::AddElementwise;
        let dtype = DType::F32;
        let elem_count = 1 << 10;
        let sc = SizeClass::from_elem_count(elem_count);
        let seed = fuel_correctness_fixtures::capture::derive_seed(op, dtype, sc);
        let input = fuel_correctness_fixtures::capture::deterministic_f32_input(
            op, 2 * elem_count,
        );
        let hash = fuel_correctness_fixtures::capture::hash_f32_input(&input);
        let fixture = CorrectnessFixture {
            op,
            dtype,
            size_class: sc,
            input_seed: seed,
            input_hash: hash,
            expected_output: vec![0u8; elem_count * 4],
            output_element_count: elem_count,
            tolerance: ToleranceBand::F32_DEFAULT,
        };
        let mut map: HashMap<(OpKind, DType, SizeClass), Vec<CorrectnessFixture>> =
            HashMap::new();
        map.insert((op, dtype, sc), vec![fixture]);
        let judge = Judge { fixtures: Some(map), ..Judge::default() };
        let got = judge.lookup_fixture(op, dtype, sc, elem_count, 2 * elem_count);
        assert!(got.is_some(), "triple-agreement fixture should be returned");
    }
}
