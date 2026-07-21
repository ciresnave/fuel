# Design — `fuel-kiss-ref-backend` + ingestion flag-not-verdict

**Date:** 2026-07-21 · **Status:** design (pending review) · **Branch:** `feat/kiss-ref-backend`
**Scope:** items #1 + #2 of the post-ratification KISS conformance-architecture follow-ups
([`docs/outreach/kiss-conformance-architecture-fuel-ratify.md`](../../outreach/kiss-conformance-architecture-fuel-ratify.md)).

> **Sibling items, for context (NOT in this spec):** #3 (transcendental-aware comparator band) is
> already done + wired in `fuel-dispatch/src/jit_ingest.rs` (`region_contains_transcendental` /
> `widen_bound_for_transcendental`). #4 (re-mint transcendental fixtures) is deferred against KISS's
> Plan B (256-bit vendored-precision core; Slice 0 = exp/log/sin f32/f64), tracked by a short KISS
> issue naming Fuel's consumer dependency.

## 1 · Purpose, the corpus reality, and the reference-selection principle

The ratified model repoints Fuel's **primitive-floor numerics** (Add/Exp/Mul/… — reference layer "(c)")
to **corpus (verdict) + kiss-ref (live diff target)**, per KISS-CONFORM §6.6-0007: *the corpus is the
authoritative Adopt/Reject verdict; kiss-ref is a differential target whose raw outcome is a signal, not
a verdict — symmetric (a discrepancy does not Reject; an agreement does not Adopt).*

**Constraint that shapes the whole design:** Fuel has **no populated corpus**. `fuel-correctness-fixtures`
is a data-model + validator only (capture tool, Judge integration, and the fixture data itself are all
*deferred* per its own docs). The KISS conformance corpus is also nascent. So the "corpus = verdict
authority" half **cannot gate anything yet** — gating Adopt on an empty corpus makes *every* input
beyond-frozen → nothing adoptable → a regression.

**Reference-selection principle (there is ONE truth — the corpus — and we pick the best available
proxy per op, not two parallel truths).** The proxy depends on the op's **determinism class**, because
that is exactly where kiss-ref *is* vs *is not* the truth:

| Op class (kiss-ref covers it) | Is kiss-ref the truth? | Reference used |
|---|---|---|
| **Exact / bitwise** (Add, Sub, Mul, Div-IEEE, integer, logical) | **Yes** — no rounding freedom; live kiss-ref === §6.5 oracle === frozen corpus cell, bit-for-bit | **kiss-ref.** Whether it *gates a verdict* or only *flags* is set by `EXACT_FLOOR_KISS_REF_GATES` — pending a §6.6-0007 ruling (precision-rule ⇒ gate; provenance-rule ⇒ flag). **Default false (advisory) until confirmed.** |
| **Transcendental** (exp/log/sin/cos/erf/rsqrt/…) | **No** — hardware-precision (libm), a genuine §6.8 ULP gap vs correctly-rounded truth | kiss-ref **advisory-flags** (item-#3 widened band); interim verdict from recipe-realize; **tight verdict awaits the wide-precision corpus** (Plan B). |
| **kiss-ref does not cover** (fused, tensor, structural, FP8, complex, any `Pending` cell) | n/a | recipe-realize (Fuel-floor) as today. |

This is the precise form of "use kiss-ref where it covers": kiss-ref becomes the *reference* for the
**exact** floor it covers (the class where it is genuinely the truth), while transcendentals stay
flag-only (the class the whole flag-not-verdict + wide-corpus machinery exists for) and uncovered ops
fall back to recipe-realize. We **select the reference per-op**; we do **not** re-plumb recipe-realize's
internals to delegate to kiss-ref (recipe-realize is transitional — the corpus replaces it — so we route
around it, not rebuild it).

kiss-ref **never** becomes a verdict source for a class where it isn't the truth (transcendentals), and
even for the exact class its verdict authority is gated on the pending §6.6-0007 ruling — so the design
honors §6.6-0007 in all cases and degrades safely (advisory) if the ruling forbids exact-op live-gating.

## 2 · Goals / Non-goals

**Goals**
- A CPU, never-panic adapter exposing kiss-ref as (a) a reference/diff-target and (b) a coverage query,
  over the stable scalar/elementwise float+int floor (kiss-ref is now **106/106** ops; we still gate every
  call on `supports()` so `Pending`/uncovered cells decline).
- Reference selection by determinism class (exact → kiss-ref; transcendental → advisory + recipe-realize;
  uncovered → recipe-realize), behind the `EXACT_FLOOR_KISS_REF_GATES` switch.
- Add `VerifyVerdict::Inconclusive` + `IngestOutcome::Flagged`; produce them faithfully; turn the current
  *hard-Fail-for-lack-of-reference* case into *Inconclusive/escalate* when kiss-ref can compare.
- A dormant `corpus_verdict` seam that later flips authority to the corpus with no `jit_ingest` re-open.
- Zero behavioural regression: existing Adopt/Reject outcomes and tests unchanged (guaranteed by the
  default-false switch + always-`None` corpus seam).

**Non-goals (explicitly deferred)**
- Populating any corpus / the capture tool (its own effort; unblocks the dormant half + #4).
- Structural/tensor/reduction/FP8/complex ops beyond what kiss-ref exposes stably (gated by `supports()`).
- Replacing recipe-realize's internals or the FKC recipe-identity check.
- Publishing kiss-ref to crates.io (see §3 — git dep now, crates.io only when kiss-ref is public-ready).
- Any GPU kernel work.

## 3 · Component #1 — `fuel-kiss-ref-backend`

New crate `fuel-kiss-ref-backend/` (new workspace member). CPU, `std`.

**Dependencies — git, not path.** kiss-ref is referenced from GitHub so cargo fetches it automatically
(no sibling checkout required, no workspace-unparseable-without-kiss-ref brittleness):
```toml
kiss-ref-core     = { git = "https://github.com/ciresnave/kiss-ref", rev = "<pinned-sha>" }
kiss-ops-vocab    = { git = "https://github.com/ciresnave/kiss-ref", rev = "<pinned-sha>" }
kiss-classify-vocab = { git = "https://github.com/ciresnave/kiss-ref", rev = "<pinned-sha>" }
```
`<pinned-sha>` is filled once the kiss-ref maintainer pushes `ciresnave/kiss-ref` (requested; local main is
at `436ff94`). Pinned-rev keeps it reproducible and cheap to bump while kiss-ref is actively growing.
**crates.io is deliberately NOT used yet** — kiss-ref stays unpublished until it's ready for public
consumption; graduate to pinned crates.io versions (baracuda-style) then. Also depends on `fuel-ir`
(`DType`, `OpTag`/`OpKind`) and `half` (`f16`/`bf16`).

**Public API (shape):**
```rust
pub enum KissRefError {           // never-panic surface; wraps kiss_ref_core::Error
    UnsupportedOp(OpTag), UnsupportedDtype(DType),
    Arity { op: OpTag, expected: usize, got: usize },
    LengthMismatch { expected: usize, got: usize },
    Eval(kiss_ref_core::Error),
}
pub fn supports(op: OpTag, dtype: DType) -> bool;              // delegates to kiss-ref `support`
pub fn is_exact_floor(op: OpTag) -> bool;                      // determinism-class split (mirror of is_transcendental)
pub fn reference_f32(op: OpTag, inputs: &[&[f32]]) -> Result<Vec<f32>, KissRefError>;   // (+ f64/f16/bf16 + int path)
pub fn diff_f32(op: OpTag, candidate: &[f32], inputs: &[&[f32]], tol: Tolerance)
    -> Result<DiffReport, KissRefError>;                       // (+ f64/f16/bf16)
```

**Mapping tables (the OpTag ↔ KISS-Ops-name gap made concrete for the floor):**
- `fn op_to_kiss(op: OpTag) -> Option<kiss_ops_vocab::Op>` — the floor subset; unmapped → `None` → decline.
- `fn dtype_to_kiss(d: DType) -> Option<kiss_classify_vocab::Dtype>` — F16/BF16/F32/F64 + ints; unmapped
  (Fuel MX, FP8) → `None` → decline.
- `is_exact_floor` classifies the covered ops into exact vs transcendental (reuses the same atom set as
  `fkc::verify::ulp::is_transcendental`, kept in sync).

**Never-panic:** every failure is a `KissRefError`; `kiss_ref_core` is itself never-panic. Satisfies
Fuel's execution-route contract (and leaves the door open to reuse the adapter as a correctness-floor
execution route later — out of scope here).

## 4 · Component #2 — ingestion flag-not-verdict

Edits confined to Fuel's lane: `fuel-dispatch/src/jit_ingest.rs` (+ its module surface). The POC's
Convergence-C does **not** touch `jit_ingest.rs` / `fkc/verify/` (confirmed).

**Outcome additions:**
- `VerifyVerdict::Inconclusive { claim: &'static str, detail: String }` — "not a verdict; escalate."
- `IngestOutcome::Flagged(FlagReport)` — claim + kiss-ref `DiffReport` summary + `escalate: true` for the
  `escalate → mint-corpus-vector` path (mint path itself: future).

**Host-data feasibility:** kiss-ref is CPU-only, so it needs the probe **inputs** and candidate **output**
on the host. Both are already materialized on the host at the comparison point (the existing
`check_numeric_bound` does its ULP compare on host slices), so kiss-ref consumes the same host buffers —
no extra D2H beyond the existing verdict path.

**Decision logic** — a pure, CPU-testable
`classify_floor_verdict(op_class, kiss: Option<DiffOutcome>, recipe_ref: Option<RefOutcome>, corpus: Option<CorpusOutcome>) -> VerifyVerdict`
(so the `#[cfg(feature="cuda")]` `verify_candidate_impl` stays a thin caller after probing). Precedence:

1. **`corpus = Some(_)`** → corpus verdict wins. *Dormant:* `corpus_verdict()` returns `None` today.
2. **Exact op, `kiss = Some(_)`** (kiss-ref is the truth here):
   - `EXACT_FLOOR_KISS_REF_GATES == true` (pending §6.6-0007) → kiss-ref **gates**: within-tolerance
     (bit-exact) → `Pass`; else → `Fail`. kiss-ref is the verdict reference for the exact floor.
   - `== false` (default until confirmed) → kiss-ref **advisory-flags** into the ledger; the verdict
     falls through to arm (4)/(5) (recipe-realize), preserving current behaviour.
3. **Transcendental op, `kiss = Some(_)`** → kiss-ref **advisory-flags** with the item-#3 widened band;
   verdict falls through to recipe-realize (interim); tight verdict awaits the corpus.
4. **`recipe_ref = Some(_)`** → existing Pass/Fail → Adopt/Reject, unchanged (item-#3 band applies). If
   kiss-ref also supported the op it already advisory-flagged in (2)/(3).
5. **`recipe_ref = None` but `kiss = Some(_)`** → *the changed case.* Today a hard
   `Fail("no decompose")` → Reject. New: kiss-ref is the only reference and (for transcendentals, or when
   the exact-gate is off) it is not authoritative → **`Inconclusive`/`Flagged`** (escalate); never Adopt
   (agreement ≠ Adopt), never hard-Reject (kiss-ref ≠ verdict). Symmetric §6.6-0007.
6. **all `None`** → unchanged current `Fail`.

**`corpus_verdict(op, dtype, inputs) -> Option<CorpusOutcome>`** — the dormant seam (consults
`fuel-correctness-fixtures`; returns `None` on the empty corpus, i.e. always today). A non-regression test
pins that it returns `None` so arm (1) stays inert.

**`ingest_one` mapping:** `Pass → Adopted`, `Fail → Rejected` (unchanged); **`Inconclusive → Flagged`**
(new), with a `ProviderFeedback::on_flagged` escalate callback distinct from adopt/reject.

## 5 · Data flow

```
candidate kernel ─▶ verify_candidate_impl (CUDA-gated: probes on device; materializes host slices)
                      │  op_class      = is_exact_floor / is_transcendental
                      │  kiss           = diff vs kiss-ref  (if supports)     ─┐
                      │  recipe_ref     = recipe-realize reference (item-#3)  ─┤
                      │  corpus         = corpus_verdict (dormant → None)     ─┤
                      ▼                                                        ▼
        classify_floor_verdict(op_class, kiss, recipe_ref, corpus) ─▶ VerifyVerdict
                      ▼
        ingest_one ─▶ Adopted | Rejected | Flagged(escalate)
```

## 6 · Testing (TDD — born-red then green, per project rule)

**`fuel-kiss-ref-backend` (CPU, no device):**
- `op_to_kiss` / `dtype_to_kiss` map the floor set; decline (→ `Unsupported*`) off it.
- `is_exact_floor`: Add/Mul/int → true; Exp/Erf → false (mirrors `is_transcendental`).
- `reference_f32(Add,…)` equals a hand-computed exact result; `Exp` matches kiss-ref within its band.
- `diff_f32`: identical slice → 0 ULP; perturbed → non-zero; length mismatch → `LengthMismatch`.
- never-panic: unsupported op/dtype returns `Err`, never unwinds.

**`jit_ingest` — via the pure `classify_floor_verdict` (CPU, no CUDA):**
- exact op, kiss within, gate **off** → verdict falls through to recipe-realize + advisory flag recorded.
- exact op, kiss within, gate **on** → `Pass` (kiss-ref gates).  exact op, kiss out, gate **on** → `Fail`.
- transcendental op, kiss out of tight band but in widened band → recipe-realize verdict stands + flag.
- `recipe_ref=None, kiss=Some` → `Inconclusive` (→ `Flagged`), NOT `Fail`, NOT `Pass`.
- `recipe_ref=Some(pass), kiss=Some(discrepant)`, gate off → still `Pass` (Adopt) + advisory flag.
- `corpus=None` → arm (1) inert; existing Pass/Fail preserved (non-regression).
- Existing Adopt/Reject unit tests unchanged and green.

**End-to-end CUDA wiring** (`#[ignore]`, live RTX 4070): a floor candidate with no decompose but kiss-ref
support drives `IngestionService` → `Flagged` with the escalate callback. Local run after the CPU suite.

**Build discipline:** `-p fuel-kiss-ref-backend` and `-p fuel-dispatch` only; one cargo at a time; CPU
tests need no live GPU. CUDA build only for the `#[ignore]` leg (long; optional).

## 7 · Risks & coordination

- **Pending §6.6-0007 ruling gates the exact-op verdict reach.** `EXACT_FLOOR_KISS_REF_GATES` defaults
  **false** (advisory) so the design is correct and non-regressing regardless of the answer; a `true`
  ruling just flips the flag + lights up arm (2)'s gate. Question is out to the KISS conformance maintainer.
- **kiss-ref must be pushed to GitHub first.** It's local-only today; the maintainer was asked to push
  `ciresnave/kiss-ref` + return a SHA to pin. The git dep is inert until that SHA exists — implementation
  of the adapter's kiss-ref calls waits on it (the outcome enum, mapping tables, and `classify_floor_verdict`
  do not, and can land first).
- **Shared-workspace manifest (coordinate before touching):** adding the crate edits **root `Cargo.toml`**
  (members + the git deps). The git dep removes the sibling-path brittleness, but the manifest edit is
  still a shared-file change — **ping the POC (`jvwnb5ut`) before editing root `Cargo.toml` and
  `fkc/mod.rs`** (both additive → trivially mergeable; sequence the merge). Work stays in the
  `feat/kiss-ref-backend` worktree.
- **Corpus dormancy is intentional, not a stub bug** — the `corpus_verdict` seam is wired + tested-inert;
  activation is a separate populate-the-corpus effort (unblocks the dormant half + #4).
- **Isolation for testability:** `classify_floor_verdict`, the mapping tables, and `is_exact_floor` are
  pure functions so the CUDA-gated `verify_candidate_impl` doesn't force a GPU for the logic tests.

## 8 · Sequencing

1. Scaffold `fuel-kiss-ref-backend` (crate + git deps + mapping tables + `supports` + `is_exact_floor`),
   TDD the pure parts. *(kiss-ref calls stubbed/gated until the pinned SHA lands.)*
2. `reference_*` / `diff_*` over the floor once the SHA is pinned, TDD.
3. Add `VerifyVerdict::Inconclusive` / `IngestOutcome::Flagged` + `classify_floor_verdict` (pure) with the
   determinism-class arms + `EXACT_FLOOR_KISS_REF_GATES` switch, TDD.
4. Wire `verify_candidate_impl` → op-class dispatch + advisory flag + the `recipe_ref=None & kiss=Some`
   → Inconclusive arm.
5. Add the dormant `corpus_verdict` seam + non-regression test.
6. Flip `EXACT_FLOOR_KISS_REF_GATES` per the §6.6-0007 ruling (one-line + arm-(2) gate test).
7. `#[ignore]` CUDA end-to-end leg (local).
8. Also lands with this branch: correct the stale item-#3 line in the ratification doc (doc-vs-code
   drift fix), since #3 is already done.
