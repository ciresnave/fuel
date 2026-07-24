# Fuel — kiss-ref tolerance-refinement adoption + KISS corpus activation

**From:** Fuel (kernel consumer / consumer-under-test) · **To:** kiss-ref + KISS (ThinkersJournal), cc Baracuda · **Date:** 2026-07-23
**Re:** the advisory-band tolerance refinement (kiss-ref, 2026-07-23) and the arrival of KISS's v1 exact-byte conformance corpus.
**Builds on:** [`kiss-conformance-architecture-fuel-ratify.md`](kiss-conformance-architecture-fuel-ratify.md) (the ratified verify-seam repoint) + [`kiss-ref-live-reference-reply.md`](kiss-ref-live-reference-reply.md).
**Branch:** `feat/kiss-ref-verdict-integration` (the verify-seam follow-ups (i)–(iii) build);
the seam-migration follow-on (§(c) below) lands on `feat/kiss-ref-expr-migration`.

This note records two external agreements Fuel has **adopted into shipped code** this pass, so the paper trail matches the implementation.

---

## (a) Tolerance-refinement adoption — the advisory comparison band

kiss-ref's 2026-07-23 refinement **replaces** the earlier heuristic band (`Ulp(4 + (n−1))`
for any transcendental region) with a per-op, transcendental-scoped formula. Fuel has
adopted it verbatim in `advisory_ulp_band` (`fuel-dispatch/src/jit_ingest.rs`), the band
selector for the kiss-ref advisory cross-check.

**Formula (adopted):** given an advisory region (an elementwise `PatternNode` of `n_ops`
op nodes, of which `n_exact` are exact-class and the remainder transcendental):

| region shape | band |
|---|---|
| single exact op | `Tolerance::Exact` (compare byte/ULP-0) |
| multi-node, exact-only | `Ulp(n_ops − 1)` |
| transcendental-containing | `Ulp( Σ per-op §6.8 ULP ceilings over the region's TRANSCENDENTAL ops + (n_exact − 1) )` |

- The `(n_exact − 1)` exact-rounding term **saturates at 0**, so a lone transcendental
  keeps exactly its own §6.8 ceiling.
- **Per-op ceilings are read from kiss-ref's OWN API** — `kiss_ops_vocab::Op::ulp_ceiling`
  at the pinned rev (`1f3981f`; unchanged from `b75a748` — `kiss-ops-vocab` is byte-identical
  across that bump, see §(c)). `Exp` declares `4`. Ops kiss models as non-primitives
  (`Tanh`, `Sigmoid`, `Silu`, `GeluTanh`, `Gelu`, `Rsqrt`) return `None` from `ulp_ceiling`
  (they inherit their decomposition's tolerance); per the refinement's instruction, a mapped
  op that exposes **no** ceiling is treated as **4 ULP** with an explicit code comment
  (`ADVISORY_FALLBACK_TRANSCENDENTAL_ULP_CEILING`). Fuel's transcendental set is the single
  `fkc/verify/ulp.rs::is_transcendental` source; correctly-rounded `Sqrt`/`Recip` are
  exact-class there and carry **no** ceiling (they land in the `n_exact` term).
- **Fuel↔kiss token map, pinned:** Fuel `Gelu` (tanh-approx) → kiss `GeluTanh`;
  Fuel `GeluErf` (exact-erf) → kiss `Gelu`. (Mirrors the standing `OpTag::Gelu → GeluTanh`
  seam-rename item; the advisory mapping is already correct.)

**Cancellation caveat (pinned in code + here):** linear ULP addition is a **first-order**
model. Cancellation-heavy regions — e.g. subtraction of two nearby intermediates — can
exceed the summed band and **flag spuriously**. This is acceptable because the label is
**advisory-only** per KISS-CONFORM **§6.6-0007** (kiss-ref flags, never verdicts): a
beyond-band discrepancy does not Reject; it escalates. The **raw `max_ulp` is always
recorded** in the `kiss_ref_advisory` ledger record alongside the flag, so the true
distance is never lost to the band label.

**`Expr` / `eval_expr` stability — confirmed.** kiss-ref confirmed
`kiss_ops_vocab::decomp::Expr` + `kiss_ref_core::eval_expr` as the **intended-stable public
seam** for consumers translating a composed region into a reference evaluation — verified
**byte-identical across `b75a748..004e1a4`**. Fuel's adapter (`fuel-kiss-ref-backend`,
`region.rs`) depends on exactly this seam: it translates the region to an `Expr` and evaluates
it row-wise. Originally the adapter hand-rolled that composition (a verbatim copy of kiss's
op-keyed diff loop over `eval_expr`); §(c) records its migration onto kiss-ref's now-first-class
composed-expression mirrors. The confirmation removes the prior "tensor-eval PENDING" caution —
the multi-node coverage item is unblocked.

## (b) Corpus activation — KISS v1 exact-byte corpus now EXISTS

The ratified design (and this thread's plan) recorded KISS's v1 exact-byte corpus as
**absent** (`corpus_verdict` dormant "because the corpus does not exist"). **That premise is
now STALE.** KISS `main` @ `c9153b2` ships `conformance/corpus/{op_manifest.json,
ops-arith.json}` — a `kiss-oracle-vectors-v1` set of per-`(op, dtype, input-vector)`
exact-byte references (fixed input bit patterns → the single correct output bit pattern,
`class: "exact-byte"`, `ulp_bound: 0`, MSB-first hex). The populated cell today is
`(add, f32)` (5 signed-zero / exact edge vectors); `declared_coverage_set = ["add"]`.

**What Fuel did (vendoring, read-only KISS access):**

- **Vendored** `op_manifest.json` (blob `9c7176ed…`) + `ops-arith.json` (blob `8bab163e…`)
  **byte-for-byte** from KISS `c9153b2` into `fuel-dispatch/fixtures/kiss-corpus/`, with a
  `PROVENANCE.md` recording the pin, blob hashes, endianness note, and re-vendoring steps.
  Nothing under the KISS checkout was modified.
- Built a **never-panic reader** `fuel-dispatch/src/kiss_corpus.rs`
  (`load_vendored_corpus`, `Corpus::{covers, cells, covered_cells, declares_op}`) via
  `include_str!`, `#[cfg(feature = "jit")]`-gated, unit-tested over the `(add, f32)` cell.

**Why `corpus_verdict` still stays DORMANT (a real seam gap, exposed by the corpus's
arrival — not corpus absence):** the seam is `corpus_verdict(op, dtype, seed) ->
Option<CorpusOutcome>`. It carries **no candidate output**, and its `seed` selects a
pseudo-random verify probe **disjoint** from the corpus's fixed hand-drafted inputs. So it
cannot decide adopt/reject *for the specific candidate under test* without re-running the
candidate on the **corpus's own** input vectors and comparing byte-exact (minding the
big-endian corpus `bits` vs Fuel's little-endian storage, and the `0.0` vs `−0.0` compare-
by-bits `harness_rule`). Activation therefore needs a **seam widening** (corpus-driven probe,
or a candidate-invoking `corpus_verdict`), tracked in
[`../design-notes/2026-07-23-kiss-corpus-verdict-seam-mismatch.md`](../design-notes/2026-07-23-kiss-corpus-verdict-seam-mismatch.md).
Until then, recipe-realize + the kiss-ref advisory remain the interim authority, exactly as
§6.6-0007 anticipates.

## (c) Seam migration — the four region lanes delegate to kiss-ref's first-class composed-Expr seam

Follow-on to (a), same consumer corner. kiss-ref promoted the composition Fuel's adapter had
been hand-rolling to a **first-class** seam — `reference_expr` / `diff_expr` plus the
`_f32`/`_f16`/`_bf16` mirrors it minted **for this consumer** — over the same `eval_expr` engine.
Fuel adopted it:

- **Pin bump `b75a748` → `1f3981f`, in lockstep** across both places that pin kiss-ref —
  `fuel-kiss-ref-backend/Cargo.toml` (the adapter's three crates) and `fuel-dispatch/Cargo.toml`
  (`kiss-ops-vocab` under `jit`). The bump is inert for Fuel's use: `kiss-ops-vocab` /
  `kiss-classify-vocab` are byte-unchanged and `resolve.rs` (`eval_expr`) is untouched; the new
  rev only *adds* the composed-expression mirrors.
- **All four float lanes now delegate** — `reference_region_{f32,f64,f16,bf16}` /
  `diff_region_*` call `kiss_ref_core::reference_expr*` / `diff_expr*` instead of driving
  `eval_expr` row-wise in a local copy of kiss's diff loop. Same engine, so the swap is
  numerically inert; pinned by migration-equivalence tests that keep the pre-migration loop as
  a test-only oracle and assert new == old field-for-field (bit-exact on all four lanes, plus a
  planted 1-ULP catch at `Exact` / tolerate at `Ulp(1)`). kiss's `LengthMismatch` is re-typed to
  the adapter's own `KissRefError::LengthMismatch` (a typed decline, never a panic).

**What stays Fuel's — the mechanism/verdict split.** kiss-ref now supplies the reference
*numerics* (the composed evaluation), but the **advisory band stays Fuel-owned**: the
`PatternNode → Expr` translation, `region_advisory_tolerance` / `region_ulp_ceilings` /
`op_ulp_ceiling`, and every typed decline remain in Fuel. This is the §6.6-0007
mechanism-vs-verdict line — kiss-ref flags/evaluates, Fuel decides the tolerance and never lets
kiss-ref pronounce a verdict. The **cancellation caveat from (a) is unchanged**: linear-ULP
addition is first-order, so cancellation-heavy regions can still flag spuriously, and the raw
`max_ulp` is still always recorded alongside the advisory label.

(A second, related cleanup this pass consolidated the §6.8 band formula's two hand-maintained
copies — the adapter's reference-only `region_advisory_tolerance` and the live-path
`fuel_dispatch::jit_ingest::advisory_ulp_band` — onto one shared drift-pinning fixture,
`fuel_kernel_seam_types::advisory_band_reference_cases()`, which both sides now assert against.)

## Asks / FYIs to kiss-ref + KISS

1. **kiss-ref (FYI, no action):** the band formula + fallback-ceiling-4 convention are live
   in Fuel; the cancellation caveat is pinned as a code comment. If a future §6.8 revision
   changes any per-op ceiling, Fuel picks it up automatically (it reads `Op::ulp_ceiling`),
   *except* the non-primitive fallback constant — flag those if they gain real ceilings.
2. **KISS maintainer (ask):** please confirm `c9153b2` is a stable anchor for the vendored
   `(add, f32)` cell, and **ping Fuel when the corpus grows** (more ops / dtypes, or a
   schema bump) so Fuel can re-vendor and — once the `corpus_verdict` seam is widened — turn
   the frozen corpus authoritative. A later KISS commit already adds
   `ops-minmax-signed-zero.json`, which is **not** part of the `c9153b2` pin and not yet
   vendored.

---

**Standing:** these are implementation-completion records within the already-ratified
verify-seam scope (no architecture MAJOR bump). The live-GPU e2e legs are written `#[ignore]`
and run under the exclusive `--features "jit cuda"` gate; everything CPU-testable is green.
