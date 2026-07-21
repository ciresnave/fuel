# fuel-kiss-ref-backend + ingestion flag-not-verdict — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CPU never-panic `fuel-kiss-ref-backend` adapter over `kiss-ref-core`, and wire kiss-ref into `fuel-dispatch` ingestion as a determinism-class-aware reference (verdict on the bitwise-exact floor, advisory flag on transcendentals) with a new `Inconclusive`/`Flagged` outcome and a dormant corpus seam.

**Architecture:** Two components. (A) `fuel-dispatch/src/jit_ingest.rs` gains the `Inconclusive`/`Flagged` outcome, a pure CPU-testable `classify_floor_verdict`, and a dormant `corpus_verdict` seam — all SHA-independent, built first. (B) a new `fuel-kiss-ref-backend` crate (git-dep on `ciresnave/kiss-ref`) providing the op/dtype mapping + `reference_*`/`diff_*`; then `verify_candidate_impl` calls it. kiss-ref is the reference only where it is the truth (bitwise-exact ops); transcendentals stay advisory + item-#3 widened band; the corpus-verdict authority is wired but inert until a corpus exists.

**Tech Stack:** Rust (edition 2024), cargo per-crate (`-p`), `kiss-ref-core`/`kiss-ops-vocab`/`kiss-classify-vocab` (git dep), `half` (f16/bf16), `#[cfg(test)]` unit tests.

**Design spec:** [`docs/superpowers/specs/2026-07-21-kiss-ref-backend-flag-not-verdict-design.md`](../specs/2026-07-21-kiss-ref-backend-flag-not-verdict-design.md).

## Global Constraints

- **Worktree only.** All work happens in the `feat/kiss-ref-backend` worktree at `C:/Projects/fuel-kiss-ref-backend`. The shared `C:/Projects/fuel` checkout's `main` is READ-ONLY. Re-fetch `origin/main` before any push.
- **Per-crate cargo only.** Never `cargo build`/`test` workspace-wide. Use `-p fuel-dispatch` / `-p fuel-kiss-ref-backend`. One cargo invocation at a time.
- **Never-panic.** Every failure is a typed `Result`/error variant. No `unwrap`/`expect`/`panic!`/indexing-that-can-panic on non-test paths.
- **TDD.** Write the failing test, RUN it and observe the expected failure, then implement, then observe green. A test that was never seen red is a defect.
- **Coordinate shared files.** Before editing the root `C:/Projects/fuel-kiss-ref-backend/Cargo.toml` (adding the member/deps), ping POC `jvwnb5ut` on claude-peers and sequence the merge. `fkc/mod.rs` is NOT touched by this plan (kept out to avoid the shared-file collision).
- **`<KISS_REF_REV>`** — the pinned git rev for `ciresnave/kiss-ref`, supplied at execution time once peer `nnb3tadk` pushes the repo (candidate: `436ff94`, the 106/106 HEAD). **Phase B (Tasks 5-8) is blocked until this exists.** Phase A (Tasks 1-4) has no such dependency — start there.
- **`EXACT_FLOOR_KISS_REF_GATES: bool = false`** by default (advisory) until KISS maintainer `44elbk9y` rules on the §6.6-0007 precision-vs-provenance question. Task 10 flips it if the ruling permits.
- **kiss-ref coverage is 106/106**; only FP8 (`e4m3`/`e5m2`) / `bool` / complex (`c32`/`c64`) dtype cells are absent — every call is gated by `supports()` so those decline cleanly.

---

## Phase A — ingestion outcome + classify (SHA-independent; `fuel-dispatch`)

### Task 1: New outcome types + the CPU-side inputs

**Files:**
- Modify: `fuel-dispatch/src/jit_ingest.rs` (enums near lines 62-83 and 301-310)
- Test: inline `#[cfg(test)]` in `fuel-dispatch/src/jit_ingest.rs`

**Interfaces:**
- Produces:
  - `pub struct FlagReport { pub entry_point: String, pub claim: &'static str, pub detail: String, pub diff_summary: Option<String>, pub escalate: bool }`
  - `IngestOutcome::Flagged(FlagReport)` (added variant; `IngestOutcome` is NOT cuda-gated)
  - `VerifyVerdict::Inconclusive { claim: &'static str, detail: String }` (added variant; `VerifyVerdict` enum def un-gated — see Step 3)
  - `ProviderFeedback::on_flagged(&self, _report: &FlagReport) {}` (default no-op)
  - `pub enum OpClass { Exact, Transcendental }`
  - `pub struct DiffOutcome { pub within: bool, pub max_ulp: Option<u64>, pub detail: String }`
  - `pub struct RefOutcome { pub pass: bool, pub claim: &'static str, pub detail: String }`
  - `pub struct CorpusOutcome { pub adopt: bool, pub claim: &'static str, pub detail: String }`

- [ ] **Step 1: Write the failing test** — append to the `#[cfg(test)] mod tests` block in `jit_ingest.rs`:

```rust
#[test]
fn new_outcome_types_construct_and_match() {
    let flag = FlagReport {
        entry_point: "k".into(),
        claim: "max_ulp",
        detail: "kiss-ref discrepancy".into(),
        diff_summary: Some("max_ulp=3".into()),
        escalate: true,
    };
    let out = IngestOutcome::Flagged(flag);
    assert!(matches!(out, IngestOutcome::Flagged(ref r) if r.escalate && r.claim == "max_ulp"));

    let v = VerifyVerdict::Inconclusive { claim: "max_ulp", detail: "no authoritative reference".into() };
    assert!(matches!(v, VerifyVerdict::Inconclusive { claim, .. } if claim == "max_ulp"));

    let d = DiffOutcome { within: false, max_ulp: Some(7), detail: "d".into() };
    let r = RefOutcome { pass: true, claim: "max_ulp", detail: "r".into() };
    let c = CorpusOutcome { adopt: true, claim: "max_ulp", detail: "c".into() };
    assert!(!d.within && d.max_ulp == Some(7) && r.pass && c.adopt);
    assert!(matches!(OpClass::Exact, OpClass::Exact));
}
```

- [ ] **Step 2: Run it, observe failure**

Run: `cargo test -p fuel-dispatch --lib new_outcome_types_construct_and_match`
Expected: FAIL — `FlagReport` / `IngestOutcome::Flagged` / `VerifyVerdict::Inconclusive` / `OpClass` / `DiffOutcome` etc. not found.

- [ ] **Step 3: Implement the types.** In `jit_ingest.rs`:
  - Add after `RejectionReport` (near line 67):

```rust
/// Escalation record for a `Flagged` ingest — kiss-ref (or another non-authoritative
/// reference) disagreed on an input beyond corpus coverage. Not a rejection: per
/// KISS-CONFORM §6.6-0007, a live kiss-ref outcome flags/escalates, never verdicts.
pub struct FlagReport {
    pub entry_point: String,
    pub claim: &'static str,
    pub detail: String,
    /// Compact kiss-ref `DiffReport` summary, if a diff was run.
    pub diff_summary: Option<String>,
    /// Always true today: this flag should escalate to mint a corpus vector.
    pub escalate: bool,
}
```

  - Add the `on_flagged` default method to `ProviderFeedback` (near line 74):

```rust
    /// A candidate was flagged for escalation (non-authoritative reference
    /// disagreement, or no authoritative reference available). Default no-op.
    fn on_flagged(&self, _report: &FlagReport) {}
```

  - Add the `Flagged` variant to `IngestOutcome` (near line 82): `Flagged(FlagReport),`
  - Remove the `#[cfg(feature = "cuda")]` line directly above `pub enum VerifyVerdict` (line ~301) so the enum type is available without CUDA (its cuda-only PRODUCERS in `verify_candidate_impl` stay gated). Add above it: `#[cfg_attr(not(feature = "cuda"), allow(dead_code))]`. Add the variant:

```rust
    /// The candidate could not be authoritatively verdicted (only a
    /// non-authoritative live reference was available, or the reference was
    /// missing): escalate, do not Adopt or Reject. §6.6-0007.
    Inconclusive { claim: &'static str, detail: String },
```

  - Add the CPU-side input types near the top of the classify region (create a new section comment after `IngestOutcome`):

```rust
/// Determinism class of a floor op — decides whether kiss-ref is the truth
/// (Exact: bit-for-bit === corpus) or a ~2×-band advisory (Transcendental).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpClass { Exact, Transcendental }

/// Fuel-side summary of a kiss-ref differential (no kiss-ref types cross here,
/// keeping `classify_floor_verdict` SHA-independent + CPU-testable).
#[derive(Debug, Clone)]
pub struct DiffOutcome { pub within: bool, pub max_ulp: Option<u64>, pub detail: String }

/// Fuel-side summary of the recipe-realize reference verdict (today's authority).
#[derive(Debug, Clone)]
pub struct RefOutcome { pub pass: bool, pub claim: &'static str, pub detail: String }

/// Fuel-side summary of a corpus verdict (dormant: `corpus_verdict` returns None).
#[derive(Debug, Clone)]
pub struct CorpusOutcome { pub adopt: bool, pub claim: &'static str, pub detail: String }
```

- [ ] **Step 4: Run tests, observe green**

Run: `cargo test -p fuel-dispatch --lib new_outcome_types_construct_and_match`
Expected: PASS. Then `cargo build -p fuel-dispatch` (no cuda) — expected: clean (only pre-existing warnings). If un-gating `VerifyVerdict` surfaces new dead-code errors, add `#[allow(dead_code)]` on the offending item; do NOT re-gate the enum.

- [ ] **Step 5: Commit**

```bash
cd C:/Projects/fuel-kiss-ref-backend
git add fuel-dispatch/src/jit_ingest.rs
git commit -m "feat(ingest): Inconclusive/Flagged outcome + CPU classify input types"
```

---

### Task 2: `classify_floor_verdict` — the determinism-class decision

**Files:**
- Modify: `fuel-dispatch/src/jit_ingest.rs`
- Test: inline `#[cfg(test)]`

**Interfaces:**
- Consumes: `OpClass`, `DiffOutcome`, `RefOutcome`, `CorpusOutcome`, `VerifyVerdict` (Task 1)
- Produces:
  - `pub const EXACT_FLOOR_KISS_REF_GATES: bool = false;`
  - `pub fn classify_floor_verdict(op_class: OpClass, kiss: Option<&DiffOutcome>, recipe: Option<&RefOutcome>, corpus: Option<&CorpusOutcome>) -> VerifyVerdict`

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn classify_corpus_wins_when_present() {
    let corpus = CorpusOutcome { adopt: true, claim: "max_ulp", detail: "corpus".into() };
    let v = classify_floor_verdict(OpClass::Exact, None, None, Some(&corpus));
    assert!(matches!(v, VerifyVerdict::Pass));
    let corpus_rej = CorpusOutcome { adopt: false, claim: "max_ulp", detail: "corpus".into() };
    let v = classify_floor_verdict(OpClass::Exact, None, None, Some(&corpus_rej));
    assert!(matches!(v, VerifyVerdict::Fail { claim, .. } if claim == "max_ulp"));
}

#[test]
fn classify_exact_gate_off_falls_through_to_recipe() {
    // EXACT_FLOOR_KISS_REF_GATES is false by default: kiss-ref is advisory,
    // recipe-realize is the verdict.
    let kiss = DiffOutcome { within: false, max_ulp: Some(5), detail: "disagree".into() };
    let recipe = RefOutcome { pass: true, claim: "max_ulp", detail: "recipe ok".into() };
    let v = classify_floor_verdict(OpClass::Exact, Some(&kiss), Some(&recipe), None);
    assert!(matches!(v, VerifyVerdict::Pass), "gate off => recipe verdict stands, kiss advisory");
}

#[test]
fn classify_transcendental_kiss_never_verdicts() {
    let kiss = DiffOutcome { within: false, max_ulp: Some(9), detail: "off".into() };
    let recipe = RefOutcome { pass: true, claim: "max_ulp", detail: "recipe ok".into() };
    let v = classify_floor_verdict(OpClass::Transcendental, Some(&kiss), Some(&recipe), None);
    assert!(matches!(v, VerifyVerdict::Pass), "transcendental kiss is advisory only");
}

#[test]
fn classify_no_reference_but_kiss_is_inconclusive() {
    // The changed case: today this is a hard Fail. Now: Inconclusive/escalate.
    let kiss = DiffOutcome { within: true, max_ulp: Some(0), detail: "agree".into() };
    let v = classify_floor_verdict(OpClass::Exact, Some(&kiss), None, None);
    assert!(matches!(v, VerifyVerdict::Inconclusive { .. }), "kiss agreement != Adopt");
    let kiss_off = DiffOutcome { within: false, max_ulp: Some(4), detail: "disagree".into() };
    let v = classify_floor_verdict(OpClass::Exact, Some(&kiss_off), None, None);
    assert!(matches!(v, VerifyVerdict::Inconclusive { .. }), "kiss discrepancy != Reject");
}

#[test]
fn classify_all_none_fails() {
    let v = classify_floor_verdict(OpClass::Exact, None, None, None);
    assert!(matches!(v, VerifyVerdict::Fail { claim, .. } if claim == "no_reference"));
}
```

- [ ] **Step 2: Run, observe failure**

Run: `cargo test -p fuel-dispatch --lib classify_`
Expected: FAIL — `classify_floor_verdict` not found.

- [ ] **Step 3: Implement**

```rust
/// Whether kiss-ref may GATE Adopt/Reject on the bitwise-exact floor (where
/// live kiss-ref === frozen corpus, bit-for-bit). Default `false` (advisory)
/// pending the KISS-CONFORM §6.6-0007 precision-vs-provenance ruling; flip to
/// `true` only if the standard permits live-gating bitwise-exact ops.
pub const EXACT_FLOOR_KISS_REF_GATES: bool = false;

/// Select the verdict for a floor-op candidate from the available references.
/// Pure + CPU-testable; the cuda `verify_candidate_impl` builds the outcomes and
/// calls this. Precedence: corpus (dormant) > exact-gate > transcendental/advisory
/// > recipe-realize > kiss-only-escalate > no-reference.
pub fn classify_floor_verdict(
    op_class: OpClass,
    kiss: Option<&DiffOutcome>,
    recipe: Option<&RefOutcome>,
    corpus: Option<&CorpusOutcome>,
) -> VerifyVerdict {
    // (1) Corpus is authoritative (dormant: corpus_verdict returns None today).
    if let Some(c) = corpus {
        return if c.adopt {
            VerifyVerdict::Pass
        } else {
            VerifyVerdict::Fail { claim: c.claim, detail: c.detail.clone() }
        };
    }
    // (2) Exact op + kiss-ref present + gating permitted => kiss-ref verdicts.
    if op_class == OpClass::Exact && EXACT_FLOOR_KISS_REF_GATES {
        if let Some(k) = kiss {
            return if k.within {
                VerifyVerdict::Pass
            } else {
                VerifyVerdict::Fail { claim: "max_ulp", detail: k.detail.clone() }
            };
        }
    }
    // (3)/(4) Recipe-realize is the interim verdict (kiss-ref, if any, was
    // advisory-flagged by the caller before this — transcendental or gate-off).
    if let Some(r) = recipe {
        return if r.pass {
            VerifyVerdict::Pass
        } else {
            VerifyVerdict::Fail { claim: r.claim, detail: r.detail.clone() }
        };
    }
    // (5) No authoritative reference, but kiss-ref could compare => escalate,
    // never Adopt (agreement != Adopt) and never hard-Reject (kiss != verdict).
    if let Some(k) = kiss {
        let detail = format!("no authoritative reference; kiss-ref {} (escalate to corpus): {}",
            if k.within { "agrees" } else { "disagrees" }, k.detail);
        return VerifyVerdict::Inconclusive { claim: "max_ulp", detail };
    }
    // (6) Nothing to compare against.
    VerifyVerdict::Fail { claim: "no_reference", detail: "no reference available".into() }
}
```

- [ ] **Step 4: Run, observe green**

Run: `cargo test -p fuel-dispatch --lib classify_`
Expected: PASS (all five).

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/jit_ingest.rs
git commit -m "feat(ingest): classify_floor_verdict with determinism-class arms + gate switch"
```

---

### Task 3: Dormant `corpus_verdict` seam

**Files:**
- Modify: `fuel-dispatch/src/jit_ingest.rs`
- Test: inline `#[cfg(test)]`

**Interfaces:**
- Produces: `pub fn corpus_verdict(op: OpKind, dtype: DType, probe_seed: u64) -> Option<CorpusOutcome>` (returns `None` on the empty corpus — always, today)

- [ ] **Step 1: Failing test**

```rust
#[test]
fn corpus_verdict_is_dormant_returns_none() {
    // Empty corpus today: every cell is uncovered, so arm (1) never fires.
    let out = corpus_verdict(OpKind::Add, fuel_ir::DType::F32, 0);
    assert!(out.is_none(), "corpus is unpopulated; verdict authority stays with recipe-realize");
}
```
(Use the real `OpKind` variant name for elementwise add — verify with `grep 'enum OpKind' fuel-ir`; substitute the actual variant if `Add` differs.)

- [ ] **Step 2: Run, observe failure**

Run: `cargo test -p fuel-dispatch --lib corpus_verdict_is_dormant_returns_none`
Expected: FAIL — `corpus_verdict` not found.

- [ ] **Step 3: Implement**

```rust
use fuel_ir::dispatch::OpKind;
use fuel_ir::DType;

/// The dormant corpus-verdict seam. When Fuel has a populated frozen corpus
/// (`fuel-correctness-fixtures` data set — deferred), a covering fixture flips
/// Adopt authority to the corpus (§6.6-0007). Until then there is no covering
/// fixture, so this returns `None` and recipe-realize stays the interim authority.
/// Wired-but-inert by design; activation needs no `jit_ingest` re-open.
pub fn corpus_verdict(_op: OpKind, _dtype: DType, _probe_seed: u64) -> Option<CorpusOutcome> {
    // No populated corpus exists; no fixture can cover any cell yet.
    None
}
```

- [ ] **Step 4: Run, observe green**

Run: `cargo test -p fuel-dispatch --lib corpus_verdict_is_dormant_returns_none`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/jit_ingest.rs
git commit -m "feat(ingest): dormant corpus_verdict seam (returns None on empty corpus)"
```

---

### Task 4: Verdict → outcome mapping (`Inconclusive → Flagged`)

**Files:**
- Modify: `fuel-dispatch/src/jit_ingest.rs` (`ingest_one`, lines 810-829)
- Test: inline `#[cfg(test)]`

**Interfaces:**
- Produces: `fn outcome_from_nonadopt_verdict(verdict: VerifyVerdict, records: Vec<LedgerRecord>, entry_point: &str) -> IngestOutcome` (non-cuda; maps `Fail → Rejected`, `Inconclusive → Flagged`; `Pass` is unreachable here and returns a defensive `Rejected`)

- [ ] **Step 1: Failing tests** (non-cuda — the mapping is pure):

```rust
#[test]
fn map_fail_verdict_to_rejected() {
    let out = outcome_from_nonadopt_verdict(
        VerifyVerdict::Fail { claim: "max_ulp", detail: "off by 3".into() }, vec![], "k");
    assert!(matches!(out, IngestOutcome::Rejected(ref r) if r.failed_claim == "max_ulp"));
}

#[test]
fn map_inconclusive_verdict_to_flagged() {
    let out = outcome_from_nonadopt_verdict(
        VerifyVerdict::Inconclusive { claim: "max_ulp", detail: "escalate".into() }, vec![], "k");
    assert!(matches!(out, IngestOutcome::Flagged(ref r) if r.escalate && r.claim == "max_ulp"));
}
```

- [ ] **Step 2: Run, observe failure**

Run: `cargo test -p fuel-dispatch --lib map_`
Expected: FAIL — `outcome_from_nonadopt_verdict` not found.

- [ ] **Step 3: Implement** the non-cuda helper, and route `ingest_one`'s non-`Pass` arms through it:

```rust
/// Map a non-`Pass` verdict to its ingest outcome. Non-cuda + pure so the
/// mapping (incl. the new `Inconclusive → Flagged` escalate path) is tested
/// without a device. `Pass` never reaches here (it adopts, which needs cuda).
fn outcome_from_nonadopt_verdict(
    verdict: VerifyVerdict,
    records: Vec<LedgerRecord>,
    entry_point: &str,
) -> IngestOutcome {
    match verdict {
        VerifyVerdict::Pass => IngestOutcome::Rejected(RejectionReport {
            entry_point: entry_point.to_string(),
            failed_claim: "internal",
            detail: "Pass routed to non-adopt mapping".into(),
            ledger_record: None,
        }),
        VerifyVerdict::Fail { claim, detail } => {
            let ledger_record = records.into_iter().find(|r| r.claim == claim);
            IngestOutcome::Rejected(RejectionReport {
                entry_point: entry_point.to_string(), failed_claim: claim, detail, ledger_record,
            })
        }
        VerifyVerdict::Inconclusive { claim, detail } => IngestOutcome::Flagged(FlagReport {
            entry_point: entry_point.to_string(), claim, detail, diff_summary: None, escalate: true,
        }),
    }
}
```

  Then change `ingest_one`'s `match verdict` (cuda) so the non-`Pass` arm delegates:

```rust
    match verdict {
        VerifyVerdict::Pass => adopt_verified(cand),
        other => outcome_from_nonadopt_verdict(other, records, &cand.entry_point),
    }
```

- [ ] **Step 4: Run, observe green + build both features**

Run: `cargo test -p fuel-dispatch --lib map_`  → PASS.
Run: `cargo build -p fuel-dispatch` (no cuda) → clean.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/jit_ingest.rs
git commit -m "feat(ingest): route Inconclusive->Flagged via pure outcome mapper"
```

---

## Phase B — the adapter crate (BLOCKED on `<KISS_REF_REV>`)

> Do not start until peer `nnb3tadk` has pushed `ciresnave/kiss-ref` and given a SHA. Substitute `<KISS_REF_REV>` everywhere below.

### Task 5: Scaffold `fuel-kiss-ref-backend` (COORDINATE root Cargo.toml)

**Files:**
- Create: `fuel-kiss-ref-backend/Cargo.toml`, `fuel-kiss-ref-backend/src/lib.rs`
- Modify: `Cargo.toml` (workspace `members`) — **PING `jvwnb5ut` FIRST**

**Interfaces:**
- Produces: crate `fuel-kiss-ref-backend`; `pub enum KissRefError { UnsupportedOp(OpTag), UnsupportedDtype(DType), Arity { op: OpTag, expected: usize, got: usize }, LengthMismatch { expected: usize, got: usize }, Eval(kiss_ref_core::Error) }`

- [ ] **Step 1: Ping the POC**, then add `"fuel-kiss-ref-backend",` to the `members` array in the root `Cargo.toml` (after `"fuel-correctness-fixtures",`). Leave it OFF `default-members` (it has a network git dep) — add it to the `default-members` list ONLY if the workspace has one that should include it; otherwise membership + `-p` is enough.

- [ ] **Step 2: Create `fuel-kiss-ref-backend/Cargo.toml`:**

```toml
[package]
name = "fuel-kiss-ref-backend"
version = "0.1.0"
edition = "2024"

[dependencies]
fuel-ir = { path = "../fuel-ir" }
half = { workspace = true }
kiss-ref-core       = { git = "https://github.com/ciresnave/kiss-ref", rev = "<KISS_REF_REV>" }
kiss-ops-vocab      = { git = "https://github.com/ciresnave/kiss-ref", rev = "<KISS_REF_REV>" }
kiss-classify-vocab = { git = "https://github.com/ciresnave/kiss-ref", rev = "<KISS_REF_REV>" }
```
(If `half` is not a workspace dep, use the version `fuel-ir` uses — check `fuel-ir/Cargo.toml`.)

- [ ] **Step 3: Create `fuel-kiss-ref-backend/src/lib.rs`** with the error type + a compile smoke test:

```rust
//! CPU, never-panic adapter exposing `kiss-ref-core` as Fuel's primitive-floor
//! reference / differential target. Correctness only; no fusion, no GPU.

use fuel_ir::DType;
use fuel_ir::dispatch::OpKind as OpTag; // adjust to Fuel's op-tag type if different

/// Never-panic failure surface. Every kiss-ref call returns `Result<_, KissRefError>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KissRefError {
    UnsupportedOp(OpTag),
    UnsupportedDtype(DType),
    Arity { op: OpTag, expected: usize, got: usize },
    LengthMismatch { expected: usize, got: usize },
    Eval(kiss_ref_core::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn error_type_constructs() {
        let e = KissRefError::LengthMismatch { expected: 4, got: 3 };
        assert!(matches!(e, KissRefError::LengthMismatch { expected: 4, got: 3 }));
    }
}
```
(Confirm Fuel's op-tag type: `grep 'OpTag' fuel-graph fuel-ir`. Use whichever type `verify_candidate` keys ops by; `OpTag` here is a placeholder for that concrete type.)

- [ ] **Step 4: Build + test**

Run: `cargo test -p fuel-kiss-ref-backend`
Expected: PASS (`error_type_constructs`); cargo fetches `ciresnave/kiss-ref` at `<KISS_REF_REV>`.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml fuel-kiss-ref-backend/
git commit -m "feat(kiss-ref-backend): scaffold crate + KissRefError (git dep on ciresnave/kiss-ref)"
```

---

### Task 6: Mapping tables (`op_to_kiss`, `dtype_to_kiss`, `is_exact_floor`)

**Files:**
- Create: `fuel-kiss-ref-backend/src/mapping.rs`
- Modify: `fuel-kiss-ref-backend/src/lib.rs` (`pub mod mapping;` + re-exports)
- Test: inline `#[cfg(test)]` in `mapping.rs`

**Interfaces:**
- Produces:
  - `pub fn op_to_kiss(op: OpTag) -> Option<kiss_ops_vocab::Op>`
  - `pub fn dtype_to_kiss(d: DType) -> Option<kiss_classify_vocab::Dtype>`
  - `pub fn is_exact_floor(op: OpTag) -> bool`
  - `pub fn supports(op: OpTag, dtype: DType) -> bool`

- [ ] **Step 1: Failing tests** (in `mapping.rs`):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::DType;
    #[test]
    fn maps_the_exact_floor() {
        assert!(op_to_kiss(OpTag::Add).is_some());
        assert!(is_exact_floor(OpTag::Add));
        assert!(dtype_to_kiss(DType::F32).is_some());
    }
    #[test]
    fn transcendental_is_not_exact() {
        assert!(op_to_kiss(OpTag::Exp).is_some());
        assert!(!is_exact_floor(OpTag::Exp));
    }
    #[test]
    fn declines_unmapped() {
        // A dtype kiss-ref lacks (FP8) declines; supports() is false.
        assert!(dtype_to_kiss(DType::F8E4M3).is_none());
        assert!(!supports(OpTag::Add, DType::F8E4M3));
    }
}
```
(Use the real Fuel `OpTag`/`DType` variant names — verify against `fuel-ir`. `F8E4M3` is illustrative; use Fuel's actual FP8 variant.)

- [ ] **Step 2: Run, observe failure**

Run: `cargo test -p fuel-kiss-ref-backend mapping`
Expected: FAIL — functions not found.

- [ ] **Step 3: Implement `mapping.rs`** — an explicit `match` for the floor subset (add/sub/mul/div/neg/recip/abs/sqrt/rsqrt/exp/log/sin/cos/tanh/erf/relu/max/min + the integer floor), mapping each `OpTag` variant to its `kiss_ops_vocab::Op`, returning `None` off the floor. `is_exact_floor` matches the exact subset (arith/int/compare/select/round/cast/bitwise) → `true`, the transcendental subset → `false`. `dtype_to_kiss` maps `F16/BF16/F32/F64` + integer dtypes, `None` for FP8/complex/bool/MX. `supports(op, dtype) = op_to_kiss(op).zip(dtype_to_kiss(dtype)).map(|(o,d)| kiss_ref_core::support(o, d) == kiss_ref_core::Support::Done).unwrap_or(false)`. (Confirm `kiss_ref_core::support`'s exact signature against the pinned rev; it returns `Support::{Done,Pending}`.)

- [ ] **Step 4: Run, observe green**

Run: `cargo test -p fuel-kiss-ref-backend mapping`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fuel-kiss-ref-backend/
git commit -m "feat(kiss-ref-backend): op/dtype mapping + is_exact_floor + supports"
```

---

### Task 7: `reference_*` / `diff_*`

**Files:**
- Create: `fuel-kiss-ref-backend/src/reference.rs`
- Modify: `fuel-kiss-ref-backend/src/lib.rs` (`pub mod reference;` + re-exports of `reference_f32/f64/f16/bf16`, `diff_f32/…`, `Tolerance`, `DiffReport`)
- Test: inline `#[cfg(test)]`

**Interfaces:**
- Produces: `pub fn reference_f32(op: OpTag, inputs: &[&[f32]]) -> Result<Vec<f32>, KissRefError>` (+ f64/f16/bf16), `pub fn diff_f32(op: OpTag, candidate: &[f32], inputs: &[&[f32]], tol: Tolerance) -> Result<DiffReport, KissRefError>` (+ f64/f16/bf16); re-exports `kiss_ref_core::{Tolerance, DiffReport}`.

- [ ] **Step 1: Failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_add_is_exact() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [10.0f32, 20.0, 30.0];
        let out = reference_f32(OpTag::Add, &[&a, &b]).unwrap();
        assert_eq!(out, vec![11.0, 22.0, 33.0]);
    }
    #[test]
    fn diff_identical_is_zero_ulp() {
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        let cand = [4.0f32, 6.0];
        let rep = diff_f32(OpTag::Add, &cand, &[&a, &b], Tolerance::Exact).unwrap();
        assert!(rep.within(), "identical output must satisfy Exact");
    }
    #[test]
    fn diff_length_mismatch_errs_not_panics() {
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        let cand = [4.0f32]; // wrong length
        assert!(matches!(diff_f32(OpTag::Add, &cand, &[&a, &b], Tolerance::Exact),
            Err(KissRefError::LengthMismatch { .. })));
    }
}
```
(Adjust `rep.within()` to the real `DiffReport` accessor at the pinned rev — check `kiss_ref_core::diff` return shape; it may be a field or a method.)

- [ ] **Step 2: Run, observe failure** — `cargo test -p fuel-kiss-ref-backend reference` → FAIL (not found).

- [ ] **Step 3: Implement `reference.rs`.** Each `reference_f32` maps the op via `op_to_kiss` (else `UnsupportedOp`), calls `kiss_ref_core::reference_f32(kiss_op, inputs)`, maps `kiss_ref_core::Error → KissRefError::Eval`. `diff_f32` computes the reference, length-checks candidate vs reference (`LengthMismatch`), then calls `kiss_ref_core::diff_f32(candidate, &reference, tol)`. (Confirm the exact `kiss_ref_core::reference_f32` / `diff_f32` signatures at `<KISS_REF_REV>` — the crate re-exports them from its `diff` module.)

- [ ] **Step 4: Run, observe green** — `cargo test -p fuel-kiss-ref-backend reference` → PASS.

- [ ] **Step 5: Commit**

```bash
git add fuel-kiss-ref-backend/
git commit -m "feat(kiss-ref-backend): reference_* + diff_* over the floor"
```

---

## Phase C — wire + finalize

### Task 8: Wire `verify_candidate_impl` to kiss-ref (cuda)

**Files:**
- Modify: `fuel-dispatch/Cargo.toml` (add `fuel-kiss-ref-backend` dep)
- Modify: `fuel-dispatch/src/jit_ingest.rs` (`verify_candidate_impl` numeric region, lines ~684-789)
- Test: inline `#[cfg(all(test, feature = "cuda"))]` `#[ignore]` live test

**Interfaces:**
- Consumes: `fuel_kiss_ref_backend::{supports, is_exact_floor, diff_f32, Tolerance}`, `classify_floor_verdict`, `corpus_verdict`, `DiffOutcome`, `OpClass` (Tasks 2-3, 6-7)

- [ ] **Step 1:** Add to `fuel-dispatch/Cargo.toml` `[dependencies]`: `fuel-kiss-ref-backend = { path = "../fuel-kiss-ref-backend" }`.

- [ ] **Step 2: Write the `#[ignore]` live-CUDA test** — a floor candidate (`Add`, f32) with `decompose: None` but kiss-ref support drives `verify_candidate` → `VerifyVerdict::Inconclusive` (not `Fail`). Model it on the existing `#[ignore]` cuda tests in the file (`verify_candidate_add_f32_passes_against_its_decompose`). Assert `matches!(verdict, VerifyVerdict::Inconclusive { .. })`.

- [ ] **Step 3: Implement the wiring** in `verify_candidate_impl`, in the numeric-claims region, computing the pieces and delegating to `classify_floor_verdict`:
  - `op_class` = `if region_contains_transcendental(dec) { OpClass::Transcendental } else { OpClass::Exact }` (reuse the item-#3 helper).
  - `kiss: Option<DiffOutcome>` = if `fuel_kiss_ref_backend::supports(op, dtype)`, run `diff_f32(op, &cand_out_host, &input_host_slices, tol)` and summarize into `DiffOutcome { within, max_ulp, detail }`; record an advisory `flag` ledger entry regardless. Else `None`.
  - `recipe: Option<RefOutcome>` = the existing recipe-realize result (Pass/Fail) expressed as `RefOutcome`, or `None` when no decompose/registered recipe exists.
  - `corpus` = `corpus_verdict(op, dtype, probe_seed)` (None today).
  - `let verdict = classify_floor_verdict(op_class, kiss.as_ref(), recipe.as_ref(), corpus.as_ref());` and return it (replacing the current hard-`Fail`-on-no-reference return with this delegation).
  - Keep the existing transcendental band-widening (item #3) applied to the `recipe`/`kiss` tolerance as today.

- [ ] **Step 4: Verify.** CPU: `cargo build -p fuel-dispatch` clean. GPU (local, optional now): `cargo test -p fuel-dispatch --features cuda -- --ignored verify_` after prepending the cuDNN path (see CLAUDE.md runtime-PATH note). Report results faithfully; if GPU unavailable, state the live leg is unrun.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/Cargo.toml fuel-dispatch/src/jit_ingest.rs
git commit -m "feat(ingest): wire kiss-ref reference into verify_candidate (advisory + escalate)"
```

---

### Task 9: End-to-end `#[ignore]` CUDA leg

**Files:** Modify: `fuel-dispatch/src/jit_ingest.rs` (test module)

- [ ] **Step 1:** Add an `#[ignore]` `#[cfg(feature = "cuda")]` test driving `IngestionService::start` with a no-decompose floor candidate → asserts an `IngestOutcome::Flagged` reaches `ProviderFeedback::on_flagged` (model on the existing Task-8 end-to-end test in the file).
- [ ] **Step 2:** Run locally if a GPU is available (`--features cuda -- --ignored`); otherwise record as unrun.
- [ ] **Step 3: Commit** `test(ingest): e2e ignore-cuda leg for Flagged escalation`.

---

### Task 10: Correct the item-#3 doc drift + (conditional) flip the gate

**Files:** Modify: `docs/outreach/kiss-conformance-architecture-fuel-ratify.md`; possibly `fuel-dispatch/src/jit_ingest.rs`

- [ ] **Step 1:** In the ratify doc §6, change item #3 from a "queued, pending" follow-up to "DONE + wired (`region_contains_transcendental`/`widen_bound_for_transcendental` in `jit_ingest.rs`)", correcting the drift. Commit `docs(outreach): correct item-#3 status (transcendental band already wired)`.
- [ ] **Step 2 (CONDITIONAL — only if `44elbk9y` ruled that §6.6-0007 permits live-gating bitwise-exact ops):** flip `EXACT_FLOOR_KISS_REF_GATES` to `true`; add a `classify_` test asserting an exact op with a discrepant kiss-ref and no recipe now `Fail`s (gates), and an in-band exact op `Pass`es. Run `cargo test -p fuel-dispatch --lib classify_`. Commit `feat(ingest): gate exact-floor verdicts on kiss-ref per §6.6-0007 ruling`. If the ruling forbids it, leave the switch `false` and record the decision in the spec.

---

## Self-Review

- **Spec coverage:** §1 reference-selection → Tasks 2, 8. §3 crate/deps/API → Tasks 5-7. §4 outcomes + classify + corpus seam + ingest map → Tasks 1-4, 8. §6 testing → each task's tests + Task 9. §7 risks (git-dep, POC coordination, gate switch, corpus dormancy) → Global Constraints + Tasks 3, 5, 10. §8 sequencing → phase order. Item-#3 doc fix → Task 10. Covered.
- **Placeholder scan:** `<KISS_REF_REV>` is an execution-time value (peer push), documented in Global Constraints — not a plan gap. `OpTag`/`DType`/`OpKind` variant names are flagged for verification against `fuel-ir`/`fuel-graph` where the concrete type differs. No TBD/TODO steps.
- **Type consistency:** `FlagReport`, `IngestOutcome::Flagged`, `VerifyVerdict::Inconclusive`, `DiffOutcome`, `RefOutcome`, `CorpusOutcome`, `OpClass`, `classify_floor_verdict`, `corpus_verdict`, `outcome_from_nonadopt_verdict`, `EXACT_FLOOR_KISS_REF_GATES`, `supports`/`op_to_kiss`/`dtype_to_kiss`/`is_exact_floor`/`reference_f32`/`diff_f32` are defined once and referenced consistently across tasks.
