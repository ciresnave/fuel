# kiss-ref verdict integration — the tracked verify-seam follow-ups (i)–(v)

**Date:** 2026-07-23 · **Thread:** `kiss-ref-verdict` · **Base:** `origin/main` @ `af4b7dd4` · **Branch:** `feat/kiss-ref-verdict` (isolated worktree)
**Scope:** ROADMAP.md:260-268 tracked follow-ups on the shipped advisory cross-check.
**Prior art (read first):** `docs/superpowers/plans/2026-07-21-kiss-ref-backend-flag-not-verdict.md` (REVISION ADDENDUM ~690-717 is authoritative) + `docs/superpowers/specs/2026-07-21-kiss-ref-backend-flag-not-verdict-design.md`.

> **For agentic workers:** use superpowers:subagent-driven-development or superpowers:executing-plans, task-by-task, TDD born-red discipline. GPU tasks run cargo FOREGROUND (background cargo deadlocks subagents — memory `gpu-subagent-foreground-cargo`).

---

## 0 · Verified grounding (re-checked 2026-07-23 against af4b7dd4 — do not re-derive, do re-verify if the base moves)

- `classify_floor_verdict` is **3-arg** `(kiss, recipe, corpus)` at `fuel-dispatch/src/jit_ingest.rs:158-195`; its only callers are tests (`flag_not_verdict_tests`, :278-374). `VerifyVerdict::Inconclusive` exists (:656) with **no producer** in `verify_candidate_impl`. `IngestOutcome::Flagged` + `ProviderFeedback::on_flagged` (:69-102) are live-dead.
- Advisory block: `jit_ingest.rs:899-947` — gated `out_dtype == F32` (:905), `single_primitive_optag` only (:906, defined :252-266), `diff_f32` with `Tolerance::Exact` vs `Ulp(4)` (`ADVISORY_TRANSCENDENTAL_ULP_CEILING = 2`, :924).
- Non-f32 numeric-claim pre-invoke hard Fail: :858-869. Numeric region: recipe-identity :1009-1024, probe-arity :1037-1054, registered-recipe realize-Err Fail :1068-1078, own-decompose realize-Err Fail :1090-1099, no-decompose Fail :1101-1111, bound checks :1127-1183.
- `outcome_from_nonadopt_verdict` :216-245 — `Inconclusive → Flagged` with `diff_summary: None` (:241). `ingest_one` :1208-1216 routes through it. Worker routes `Flagged → on_flagged` :1455-1459 — **no test exercises this at service level** (grep: on_flagged appears only at 85/91/1457).
- Adapter crate `fuel-kiss-ref-backend` (already a workspace member, git dep pinned `b75a748f…` in its Cargo.toml): `mapping.rs` (`op_to_kiss` ~29 tags, `dtype_to_kiss`, `supports`), `reference.rs` (`reference_/diff_{f32,f64,f16,bf16}` via macro, private `to_rows`). Deps: `fuel-ir`, **`fuel-kernel-seam-types`** (PatternNode/OpTag/OpAttrs live there — `fuel_graph::jit` re-exports them, fuel-graph/src/jit.rs:16), `half`, the three kiss crates.
- **kiss-ref @ b75a748** (checkout `~/.cargo/git/checkouts/kiss-ref-4d1b800554dbb2a8/b75a748`): `kiss_ref_core::eval_expr(e: &Expr, inputs: &[T]) -> Result<T, Error>` (resolve.rs:132, exported lib.rs:61) over `kiss_ops_vocab::decomp::Expr { Input(u8), Const(ConstSym), Apply(Op, Vec<Expr>) }` (decomp.rs:102-110). `ulp_distance_{f32,f64,f16,bf16}` + `DiffReport { n, mismatches, max_ulp, first_mismatch }` + `conforms()` + `Tolerance::{Exact, Ulp}` all exported (diff.rs). `ScalarFloat` covers f32/f64/f16/bf16. **The 2026-07-17 “tensor-eval PENDING” caution is stale — item (iv) is NOT blocked.**
- **KISS v1 exact-byte corpus does NOT exist**: `C:\Projects\KISS\conformance` has no `corpus/` dir; kiss-ref-conformance has no corpus reader. → `corpus_verdict` stays dormant (design D4).
- f64/f16/bf16 plumbing all exists: `element_kind_to_dtype` (jit_adopt.rs:40-52), `to_bytes` (fkc/verify/harness.rs:176-193), `CudaInvoker` allocates by `out_dtype.size_in_bytes()` (invoker_cuda.rs:61-62), `baracuda_dispatch::binary::{add_f64,add_f16,add_bf16}` exist + registered (baracuda_dispatch.rs:2431-2449, 3576-3579). `PrecisionGuarantee::REFERENCE` = bit_stable + max_ulp Some(0) (fused.rs:148-154).
- `runtime_region(id) -> Option<PatternNode>` — fuel-graph/src/runtime_fused.rs:202 (None for static ids; `.read().unwrap()` inside — see Risks). `region_contains_transcendental` — fkc/verify/ulp.rs:124, **unconditionally compiled** (only jit_ingest’s import of it is cuda-gated today, :19-24 — move it to the un-gated import).
- Feature topology: `jit_ingest` is `#[cfg(feature = "jit")]` (lib.rs:58-66); `fuel-kiss-ref-backend` is **optional, enabled only by `cuda`** (fuel-dispatch/Cargo.toml:19,77); `default = []`. ⇒ CPU gate is `--features jit`; live gate is `--features "jit cuda"`. **CPU-testable helpers must not reference adapter types.**

## 1 · Global constraints

- **Worktree only** (`feat/kiss-ref-verdict`); `C:/Projects/fuel` main is read-only. `jit_ingest.rs` is exclusively this thread’s file — no cross-thread collision expected, but re-fetch origin/main before push.
- **Per-crate cargo, one invocation at a time.** Never workspace-wide. GPU legs: exclusive, foreground, `--test-threads=1`.
- **Never-panic:** all new paths return typed errors; adapter’s new variants keep `KissRefError` the single failure surface. `catch_unwind` boundaries unchanged.
- **TDD:** every task’s red test is observed red before implementation (exceptions are explicitly labeled “pin, expected born-green” — there is exactly one, T6a).
- **Behavioral non-regression pins:** `kiss_ref_advisory_records_for_add_f32` (:379-422), `verify_candidate_add_f32_passes_against_its_decompose`, both Task-8 e2e legs (:2676-2756), the whole CPU classify/map/corpus suite.

## 2 · Design decisions (settled — see the structured summary for rationale)

- **D1 (ii):** FusedOpId→advisory = **derive-from-registry**: `advisory_region = cand.decompose.cloned().or_else(|| cand.claimed_op.and_then(runtime_region))`. No static table. Static ids decline (non-elementwise anyway).
- **D2 (iv):** multi-node advisory = **PatternNode→`Expr` translation in the adapter** + row-wise `eval_expr`. Elementwise, attrs==default, mapped ops only; `SeeThrough`/`Any` decline. New `KissRefError::{UnsupportedAttrs(OpTag), UnsupportedNode}`.
- **D3:** advisory band: single exact op → `Exact`; transcendental region → `Ulp(4 + (n−1))`; multi-node exact → `Ulp(n−1)`; raw `max_ulp` always recorded.
- **D4 (i):** Inconclusive **only** when a kiss diff exists and the recipe reference is unusable: realize-Err arms, and the non-f32-numeric-claim coverable case. Hard Fails unchanged: probe/invoke/no_guarantee/recipe_identity/probe_arity/bit-stability/no-identity. `Inconclusive.claim = "max_ulp"` (names the evidence).
- **D5:** corpus stays **dormant** (verified absent) but precedence goes live: consult `corpus_verdict(single_primitive_optag(region), out_dtype, seed)` first in the numeric region; multi-node ⇒ corpus=None.
- **D6 (iii):** dtype dispatch via cuda-gated `run_region_diff` matching F32/F64/F16/BF16 + `bytes_to_{f64,f16,bf16}`; coverage via adapter `region_supported(region, dtype)`.
- **D7 (v):** live Flagged e2e = the **f64 add candidate** (born-red today at the f32-only Fail). CPU service test = routing pin (born-green, labeled).
- **D8:** `outcome_from_nonadopt_verdict` threads a compact kiss summary into `FlagReport.diff_summary` from the `kiss_ref_advisory` record.

## 3 · Tasks

### T1 — Adapter region evaluator (CPU) — `fuel-kiss-ref-backend`
**Files:** create `src/region.rs`; modify `src/lib.rs` (module + re-exports + 2 new `KissRefError` variants); `src/reference.rs` (`to_rows` → `pub(crate)`).
**API:**
```rust
pub fn region_supported(region: &PatternNode, dtype: DType) -> bool;   // every Op node mapped ∧ Support::Done ∧ attrs==default; ≥1 Op node
pub fn region_op_count(region: &PatternNode) -> usize;                 // Op nodes only
pub fn reference_region_f32(region: &PatternNode, operands: &[&[f32]]) -> Result<Vec<f32>, KissRefError>;  // + f64/f16/bf16
pub fn diff_region_f32(region: &PatternNode, candidate: &[f32], operands: &[&[f32]], tol: Tolerance) -> Result<DiffReport, KissRefError>;  // + f64/f16/bf16
```
Internals: `fn region_to_expr(&PatternNode) -> Result<Expr, KissRefError>` (`Bind{i}`→`Input(i)`; `Op`→ mapped + attrs-default guard → `Apply`; else decline). `diff_region_*` builds `DiffReport` directly (fields are pub) using `kiss_ref_core::ulp_distance_*` — kiss’s `diff_*` is op-keyed and can’t take an Expr; replicate its loop semantics exactly (both-NaN 0 / one-NaN MAX / Exact vs Ulp(n)).
**Steps:** red test `region_relu_add_matches_hand_math` (+ the decline/ragged/narrow-dtype suite listed in the task table) → observe RED (compile) → implement → `cargo test -p fuel-kiss-ref-backend` green → commit `feat(kiss-ref-backend): region evaluator (PatternNode→Expr, reference/diff over composed regions)`.

### T2 — CPU helpers in `jit_ingest.rs`
- `bytes_to_f64/bytes_to_f16/bytes_to_bf16` mirroring `bytes_to_f32` (:271-276).
- `fn advisory_ulp_band(region: &PatternNode) -> Option<u64>` per D3 (uses `region_contains_transcendental` — **move its import off the cuda-gated block**; count op nodes locally, no adapter types).
- `fn advisory_region(decompose: Option<&PatternNode>, claimed: Option<FusedOpId>) -> Option<PatternNode>` per D1.
**Red tests:** `advisory_ulp_band_selects_by_region_shape`, `advisory_region_resolves_runtime_claimed_id_and_declines_static` (register a runtime Add region; assert ROPE + unregistered id → None), byte round-trips. RED (not found) → green under `cargo test -p fuel-dispatch --features jit --lib`. Commit.

### T3 — Generalize the advisory block (:899-947) (cuda code; CPU-clean build)
Replace the f32/single-primitive advisory with:
```rust
let advisory = advisory_region(cand.decompose.as_ref(), cand.claimed_op);
let mut kiss_outcome: Option<DiffOutcome> = None;
if let Some(region) = &advisory {
    if matches!(out_dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16)
        && fuel_kiss_ref_backend::region_supported(region, out_dtype) {
        let band = advisory_ulp_band(region);
        let tol = band.map_or(Tolerance::Exact, Tolerance::Ulp);
        if let Ok(report) = run_region_diff(region, out_dtype, &cand_out.bytes, &probe, tol) {
            kiss_outcome = Some(DiffOutcome { within: report.conforms(), max_ulp: Some(report.max_ulp), detail: /* compact */ });
            ledger.upsert(make_record("kiss_ref_advisory", if report.conforms() { "pass" } else { "flag" },
                json!({ "dtype": …, "op_count": …, "max_ulp": report.max_ulp, "mismatches": report.mismatches,
                        "advisory_band_ulp": band, "source": if cand.decompose.is_some() { "decompose" } else { "claimed_recipe" },
                        "note": "advisory only; kiss-ref flags, never verdicts (§6.6-0007)" })));
        }
    }
}
```
`run_region_diff` = dtype match → `bytes_to_*` (candidate + each probe column) → `diff_region_*`. Keep the record claim/evidence keys the existing test asserts. Write the two live `#[ignore]` born-red tests here (run in T6): `multi_node_region_advisory_flags_add_kernel_for_relu_add`, `claimed_op_candidate_reaches_advisory_via_runtime_region`. Gate: `cargo build -p fuel-dispatch --features "jit cuda"` clean; CPU suite still green. Commit.

### T4 — Live classify wiring + diff_summary (cuda + one CPU-red)
1. **CPU red:** `map_inconclusive_carries_diff_summary_from_advisory_record` → extend `outcome_from_nonadopt_verdict` (D8). Green.
2. **cuda:** in the numeric region — corpus consult first (D5; a `Some` returns via `classify_floor_verdict(None, None, corpus)`); the two realize-Err arms (:1068-1078, :1090-1099) and the no-decompose arm (:1101-1111) now build on `kiss_outcome` and delegate: `classify_floor_verdict(kiss_outcome.as_ref(), None, None)` — Inconclusive when kiss present (ledger record result `"inconclusive"`, realize error in evidence), the same Fail as today when not. Happy path (bound checks :1127-1183) unchanged — it IS classify arm (2) with corpus already known-None; state this equivalence in a comment.
Gates: CPU suite green; cuda build clean. Commit.

### T5 — Non-f32 numeric-claim escalate (cuda)
Restructure :858-869: compute advisory eligibility pre-invoke (`advisory_region` + `region_supported(region, out_dtype)` + dtype ∈ {F64,F16,BF16}); ineligible ⇒ **identical** early Fail (same claim/detail bytes); eligible ⇒ proceed (invoke → advisory block → bit-stability). In the numeric region, after `recipe_identity`/`probe_arity` still gate, the escalate path skips realize + f32 bound checks and returns `classify_floor_verdict(kiss_outcome.as_ref(), None, corpus.as_ref())` ⇒ Inconclusive. CPU red: `nonf32_escalate_eligible` predicate tests. Live `#[ignore]` red: `verify_candidate_add_f64_is_inconclusive_not_failed` (add_f64 + Add decompose + REFERENCE ⇒ today `Fail{"max_ulp", …f32-only…}`). Commit.

### T6 — Service e2e (v) + the live-GPU pass
- CPU: extend `RecordingFeedback` with a `flagged` vec + `on_flagged`; `worker_routes_flagged_to_on_flagged` via `start_with_verify(|_| IngestOutcome::Flagged(…))` — **pin, expected born-green** (say so in the doc comment).
- Live `#[ignore]` red: `ingestion_service_flags_an_f64_add_candidate_e2e` — `IngestionService::start` + the T5 candidate (distinct entry_point/hash per the process-global-registration discipline, model on :2676-2714); extend `E2eFeedback` with `flagged` + `on_flagged`; assert on_flagged fired (escalate, Some diff_summary), on_rejected/on_adopted empty.
- **Live-GPU verification pass (this task owns the GPU leg):** VS dev shell or `NVCC_CCBIN` set; FOREGROUND, exclusive:
  `cargo test -p fuel-dispatch --features "jit cuda" -- --ignored kiss_ verify_candidate_ ingestion_service_ multi_node_ claimed_op_candidate_ --test-threads=1`
  Observe the T3/T5/T6 tests transition red→green (run once BEFORE the wiring lands if sequencing allows, else rely on the recorded per-task red observations); pre-existing legs stay green. Report the actual output; if the GPU is unavailable, say plainly the live leg is unrun and stop short of claiming done. Commit.

### T7 — Docs (same change)
ROADMAP.md:260-268: follow-ups (i)–(v) → shipped, with residual gaps named (static-op advisory pending PatternNode-data migration; corpus dormant pending KISS v1 corpus; non-f32 recipe VERDICT still f32-only — escalate only). `docs/outreach/kiss-conformance-architecture-fuel-ratify.md` §6 status lines updated to match. No architecture MAJOR bump (implementation completion of ratified scope). Commit `docs: mark verify-seam follow-ups (i)-(v) shipped; name residual gaps`.

## 4 · Gates (exact commands)

| # | Command | CPU/GPU |
|---|---|---|
| 1 | `cargo test -p fuel-kiss-ref-backend` | CPU |
| 2 | `cargo build -p fuel-dispatch` | CPU |
| 3 | `cargo test -p fuel-dispatch --features jit --lib` | CPU |
| 4 | `cargo build -p fuel-dispatch --features "jit cuda"` | GPU toolchain (build) |
| 5 | `cargo test -p fuel-dispatch --features "jit cuda" -- --ignored kiss_ verify_candidate_ ingestion_service_ multi_node_ claimed_op_candidate_ --test-threads=1` | GPU live (exclusive, foreground) |

## 5 · Risks (summary — full list in the structured output)
Multi-node band heuristic may flag legitimate FMA-contracting kernels (advisory-only; tune later). `runtime_region` panics on a poisoned lock (swallowed by the verify-path catch_unwind; fuel-graph fix is out of file-set). T5 must keep uncoverable-non-f32 Fail bytes identical. Dual-feature compile drift (gates 2/3/4 cover). Live legs need the VS-dev-shell/NVCC_CCBIN CUDA build environment; fuel-dispatch needs no cuDNN PATH prepend. Pin-rev is b75a748f — re-verify kiss API shapes on any bump (`Op` is non_exhaustive). f16/bf16 advisory may flag kernels that compute in f32 and round once (advisory-only; live pin uses the bit-exact f64 Add).

## 6 · External coordination (drafted, orchestrator sends — non-blocking)
1. kiss-ref peer: Expr/eval_expr stability ask + band-heuristic comment ask + region-flag→per-op-corpus-vector FYI.
2. KISS maintainer: corpus-still-absent status note; ping-when-merged request. Neither blocks the build.

