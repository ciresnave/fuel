# Design — `fuel-kiss-ref-backend` + ingestion flag-not-verdict

**Date:** 2026-07-21 · **Status:** design (pending review) · **Branch:** `feat/kiss-ref-backend`
**Scope:** items #1 + #2 of the post-ratification KISS conformance-architecture follow-ups
([`docs/outreach/kiss-conformance-architecture-fuel-ratify.md`](../../outreach/kiss-conformance-architecture-fuel-ratify.md)).

> **Sibling items, for context (NOT in this spec):** #3 (transcendental-aware comparator band) is
> already done + wired in `fuel-dispatch/src/jit_ingest.rs` (`region_contains_transcendental` /
> `widen_bound_for_transcendental`). #4 (re-mint transcendental fixtures) is deferred against KISS's
> Plan B (256-bit vendored-precision core; Slice 0 = exp/log/sin f32/f64), tracked by a short KISS
> issue naming Fuel's consumer dependency.

## 1 · Purpose & the corpus reality

The ratified model repoints Fuel's **primitive-floor numerics** (Add/Exp/Mul/… — reference layer "(c)")
to **corpus (verdict) + kiss-ref (live diff target)**, per KISS-CONFORM §6.6-0007: *the corpus is the
authoritative Adopt/Reject verdict; kiss-ref is a differential target whose raw outcome is a signal, not
a verdict — symmetric (a discrepancy does not Reject; an agreement does not Adopt).*

**Constraint that shapes the whole design:** Fuel has **no populated corpus**. `fuel-correctness-fixtures`
is a data-model + validator only — its own docs list the capture tool, the Judge integration, and the
actual fixture data as *deferred*. The KISS conformance corpus is also nascent (Plan A just landed). So
the "corpus = verdict authority" half **cannot gate anything yet** — gating Adopt on an empty corpus
would make *every* input beyond-frozen → nothing adoptable → a regression of the existing ingest path.

Therefore this increment delivers:

- **#1 — full value:** `fuel-kiss-ref-backend`, the scalar/elementwise-floor adapter over `kiss-ref-core`.
- **#2 — two halves, one live:**
  - **Live:** the `Inconclusive`/`Flagged` outcome + kiss-ref wired as a **discrepancy-detector** that
    *flags/escalates, never verdicts* (faithful §6.6-0007: kiss-ref is never a verdict source).
  - **Dormant:** a `corpus_verdict(...)` seam that returns `None` today (empty corpus) and, once a corpus
    exists, supersedes the interim authority. Wired but inert — no behaviour change now.

**Interim verdict authority stays the existing recipe-realize path** (`reference_from_registered_recipe`
/ `reference_output`) — Fuel's own floor realize, with item #3's transcendental band. kiss-ref is *added
alongside* as an independent advisory cross-check; it never replaces the verdict and never rejects.

## 2 · Goals / Non-goals

**Goals**
- A CPU, never-panic adapter exposing kiss-ref as (a) a reference/diff-target and (b) a coverage query,
  over the stable scalar/elementwise float+int floor.
- Add `VerifyVerdict::Inconclusive` + `IngestOutcome::Flagged` and produce them faithfully.
- Turn the current *hard-Fail-for-lack-of-reference* case into an *Inconclusive/escalate* case when
  kiss-ref can serve as a (non-authoritative) floor comparison.
- Record kiss-ref agreement/disagreement as advisory ledger entries on the existing verdict path.
- A dormant `corpus_verdict` seam that later flips authority to the corpus with no `jit_ingest` re-open.
- Zero behavioural regression: existing Adopt/Reject outcomes and tests unchanged.

**Non-goals (explicitly deferred)**
- Populating any corpus / building the capture tool (its own effort; unblocks the dormant half + #4).
- Structural/tensor/reduction/FP8/complex ops — kiss-ref reports these `Pending`; bind them when its
  in-flight tensor-eval layer lands.
- Replacing the recipe-realize reference or the FKC recipe-identity check.
- Any GPU kernel work.

## 3 · Component #1 — `fuel-kiss-ref-backend`

New crate `fuel-kiss-ref-backend/` (new workspace member). CPU, `std`.

**Dependencies:** `kiss-ref-core`, `kiss-ops-vocab`, `kiss-classify-vocab` (sibling path deps to
`C:/Projects/kiss-ref/crates/*`, mirroring the existing aocl/vulkane sibling-path pattern), plus
`fuel-ir` (for `DType`, `OpTag`/`OpKind`). See §7 (Risks) for the workspace-membership caveat.

**Public API (shape):**
```rust
pub enum KissRefError {           // never-panic surface; wraps kiss_ref_core::Error
    UnsupportedOp(OpTag),
    UnsupportedDtype(DType),
    Arity { op: OpTag, expected: usize, got: usize },
    LengthMismatch { expected: usize, got: usize },
    Eval(kiss_ref_core::Error),
}

/// Is (op, dtype) in kiss-ref's `Done` set (a live diff target)?  Delegates to kiss-ref `support`.
pub fn supports(op: OpTag, dtype: DType) -> bool;

/// kiss-ref's reference output for a floor op over typed host inputs (the diff target).
pub fn reference_f32(op: OpTag, inputs: &[&[f32]]) -> Result<Vec<f32>, KissRefError>;
pub fn reference_f64(op: OpTag, inputs: &[&[f64]]) -> Result<Vec<f64>, KissRefError>;
// f16 / bf16 via half::{f16,bf16}; integer path via reference_int over kiss-ref's int floor.

/// Differential report of a candidate vs kiss-ref's reference, under a tolerance.
pub fn diff_f32(op: OpTag, candidate: &[f32], inputs: &[&[f32]], tol: Tolerance)
    -> Result<DiffReport, KissRefError>;   // (+ f64/f16/bf16)
```

**Mapping tables (the crux — this is the OpTag ↔ KISS-Ops-name gap made concrete for the floor):**
- `fn op_to_kiss(op: OpTag) -> Option<kiss_ops_vocab::Op>` — the scalar/elementwise floor subset:
  Add, Sub, Mul, Div, Neg, Recip, Abs, Exp, Log, Sqrt, Rsqrt, Sin, Cos, Tanh, Erf, Relu, Max, Min,
  (+ the int floor ops). Unmapped → `None` → `UnsupportedOp` (decline, never guess).
- `fn dtype_to_kiss(d: DType) -> Option<kiss_classify_vocab::Dtype>` — F16/BF16/F32/F64 + the integer
  dtypes kiss-ref implements. Unmapped (e.g. Fuel MX dtypes, FP8) → `None` → `UnsupportedDtype`.

**Never-panic:** every failure is a `KissRefError`; `kiss_ref_core` is itself never-panic, so no path
unwinds. This satisfies Fuel's execution-route contract, letting the same adapter double as a future
correctness-floor execution route (out of scope here, but the API doesn't preclude it).

## 4 · Component #2 — ingestion flag-not-verdict

Edits confined to Fuel's lane: `fuel-dispatch/src/jit_ingest.rs` (+ its module surface). The POC's
Convergence-C does **not** touch `jit_ingest.rs` / `fkc/verify/` (confirmed).

**Outcome additions:**
- `VerifyVerdict::Inconclusive { claim: &'static str, detail: String }` — "not a verdict; escalate."
- `IngestOutcome::Flagged(FlagReport)` — carries the claim + the kiss-ref `DiffReport` summary + an
  `escalate: true` marker for the `escalate → mint-corpus-vector` path (mint path itself: future).

**Host-data feasibility:** kiss-ref is CPU-only, so it needs the probe **inputs** and the candidate
**output** on the host. Both are already materialized on the host at the comparison point — the existing
`check_numeric_bound` does its ULP compare on host slices — so kiss-ref consumes the same host buffers;
no extra D2H beyond what the existing verdict path already performs.

**Decision logic** (extracted into a pure, CPU-testable function
`classify_floor_verdict(recipe_ref: Option<RefOutcome>, kiss: Option<DiffOutcome>, corpus: Option<CorpusOutcome>) -> VerifyVerdict`
so it is unit-testable without a `CudaDevice`; the `#[cfg(feature="cuda")]` `verify_candidate_impl`
calls it after probing):

1. **`corpus = Some(_)`** → the corpus verdict wins (Adopt/Reject). *Dormant:* `corpus_verdict()`
   returns `None` today, so this arm never fires yet.
2. **`recipe_ref = Some(_)`** (today's authority) → existing Pass/Fail → Adopt/Reject, unchanged
   (item #3 band still applies). **If kiss-ref supports the op, additionally run `diff` and record an
   advisory `flag` ledger entry** (agreement or discrepancy) — advisory only, does **not** change the
   verdict. This is the independent floor cross-check.
3. **`recipe_ref = None` but `kiss = Some(_)`** → *the changed case.* Today this is a hard
   `Fail("no decompose: cannot verify")` → Reject. New: kiss-ref is the only reference and it is not
   authoritative, so → **`Inconclusive`/`Flagged`** (escalate), never Adopt (agreement ≠ Adopt) and
   never hard-Reject (kiss-ref ≠ verdict). Symmetric §6.6-0007, faithfully.
4. **all `None`** → unchanged current `Fail`.

**`corpus_verdict(op, dtype, inputs) -> Option<CorpusOutcome>`** — the dormant seam. Consults
`fuel-correctness-fixtures` (`validate_against_fixture` when a covering fixture is found). Returns `None`
whenever no fixture covers the cell — which is *always*, today. A non-regression test pins that it
returns `None` on the empty corpus so arm (1) stays inert.

**`ingest_one` mapping:** `Pass → Adopted`, `Fail → Rejected` (unchanged); **`Inconclusive → Flagged`**
(new), with a `ProviderFeedback::on_flagged` escalate callback distinct from adopt/reject.

## 5 · Data flow

```
candidate kernel ─▶ verify_candidate_impl (CUDA-gated: probes on device)
                      │  recipe-realize reference (existing, item-#3 band)  ─┐
                      │  kiss-ref diff  (fuel-kiss-ref-backend, if supports) ─┤
                      │  corpus_verdict (dormant → None)                     ─┤
                      ▼                                                       ▼
              classify_floor_verdict(recipe_ref, kiss, corpus) ─▶ VerifyVerdict
                      ▼
              ingest_one ─▶ Adopted | Rejected | Flagged(escalate)
```

## 6 · Testing (TDD — born-red then green, per project rule)

**`fuel-kiss-ref-backend` (CPU, no device):**
- `op_to_kiss` / `dtype_to_kiss` map the floor set; decline (→ `Unsupported*`) off it.
- `reference_f32(Add, …)` etc. equals a hand-computed floor result; a transcendental (`Exp`) matches
  kiss-ref within its declared band.
- `diff_f32`: identical slice → 0 ULP; perturbed slice → non-zero; length mismatch → `LengthMismatch`.
- never-panic: unsupported op/dtype returns `Err`, never unwinds.

**`jit_ingest` (CPU-testable via the extracted `classify_floor_verdict`):**
- `recipe_ref=None, kiss=Some(within)` → `Inconclusive` (→ `Flagged`), NOT `Fail`, NOT `Pass`.
- `recipe_ref=None, kiss=Some(discrepant)` → `Inconclusive`/`Flagged` (escalate), NOT hard `Reject`.
- `recipe_ref=Some(pass), kiss=Some(discrepant)` → still `Pass` (Adopt) + advisory flag in ledger.
- `corpus=None` (empty corpus) → arm (1) inert; existing Pass/Fail preserved (non-regression).
- Existing Adopt/Reject unit tests unchanged and green.

**End-to-end CUDA wiring** (`#[ignore]`, live RTX 4070): a floor candidate with no decompose but
kiss-ref support drives `IngestionService` → `Flagged` (not Rejected) with the escalate callback fired.
Run locally after the CPU suite is green; not required for the CPU TDD gate.

**Build discipline:** `-p fuel-kiss-ref-backend` and `-p fuel-dispatch` only; one cargo at a time; the
CPU tests need no live GPU. CUDA build only for the `#[ignore]` leg (long; optional).

## 7 · Risks & coordination

- **Shared-workspace manifest (coordinate before touching):** adding `fuel-kiss-ref-backend` edits the
  **root `Cargo.toml`** members list, and adds `kiss-ref` as a **required sibling path dep** (a new hard
  requirement for anyone parsing the workspace — like aocl/vulkane). Two mitigations to decide at
  implementation: keep it out of `default-members` so a bare `-p other` is unaffected *where possible*
  (note: cargo still parses all member manifests, so kiss-ref must be present); document the new sibling
  in the environment-discipline note. **Ping the POC (`jvwnb5ut`) before editing root `Cargo.toml` and
  `fkc/mod.rs`** (both shared; additive → trivially mergeable, but sequence the merge).
- **kiss-ref coverage is 78/106 and its tensor-eval layer is in flux.** We bind only the stable
  scalar/elementwise floor; `supports()` gates every call, so an unimplemented cell simply declines.
- **Corpus dormancy is intentional, not a stub bug.** The `corpus_verdict` seam is wired + tested-inert;
  activation is a separate populate-the-corpus effort. Documented as such so it is not mistaken for a
  shipped-but-unwired defect.
- **Isolation for testability:** `classify_floor_verdict` and the mapping tables are pure functions so
  the CUDA-gated `verify_candidate_impl` doesn't force a GPU for the logic tests.

## 8 · Sequencing

1. Scaffold `fuel-kiss-ref-backend` (crate + deps + mapping tables + `supports`), TDD the adapter.
2. `reference_*` / `diff_*` over the floor, TDD.
3. Add `VerifyVerdict::Inconclusive` / `IngestOutcome::Flagged` + `classify_floor_verdict` (pure), TDD.
4. Wire `verify_candidate_impl` → advisory flag + the `recipe_ref=None & kiss=Some` → Inconclusive arm.
5. Add the dormant `corpus_verdict` seam + non-regression test.
6. `#[ignore]` CUDA end-to-end leg (local).
7. Also lands with this branch: correct the stale item-#3 line in the ratification doc (doc-vs-code
   drift fix), since #3 is already done.
