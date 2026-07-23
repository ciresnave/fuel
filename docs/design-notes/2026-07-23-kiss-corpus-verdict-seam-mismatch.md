# KISS corpus exists — but `corpus_verdict`'s seam can't consume it (yet)

**Date:** 2026-07-23 · **Task:** A4b (kiss-ref verdict integration) · **Branch:** `feat/kiss-ref-verdict-integration`
**Status:** corpus vendored + reader built; `corpus_verdict` left DORMANT per the A4b best-effort guard.

## TL;DR

KISS's v1 exact-byte golden corpus **now exists** (the plan's "corpus verified
absent" premise is stale). It is vendored at `fuel-dispatch/fixtures/kiss-corpus/`
(KISS `main` @ `c9153b2`) and parsed by `fuel-dispatch/src/kiss_corpus.rs`.

But it does **not** map onto `jit_ingest::corpus_verdict`'s current signature as a
candidate verdict, so `corpus_verdict` stays dormant (`None`). This note records
why, and the seam shape that would activate it.

## What the corpus is

`ops-arith.json` (schema `kiss-oracle-vectors-v1`) is a list of test vectors.
Each vector is a per-`(op, dtype, input-vector)` **exact-byte reference**:

```json
{"tcId": 4, "op": "add", "dtype": "f32",
 "inputs": [{"role":"a","bits":"3F 80 00 00"}, {"role":"b","bits":"3F 80 00 00"}],
 "expected": {"bits":"40 00 00 00"}, "class": "exact-byte", "ulp_bound": 0}
```

i.e. fixed input bit patterns → the single correct output bit pattern. It is an
**oracle** (reference outputs for chosen inputs), NOT an `(op, dtype) →
adopt/reject` verdict table. `op_manifest.json`'s `declared_coverage_set` is
`["add"]`; the only populated cell today is `(add, f32)` with 5 signed-zero /
exact edge vectors.

## Why it doesn't fit `corpus_verdict(op, dtype, seed)`

The seam is:

```rust
pub fn corpus_verdict(op: OpTag, dtype: DType, seed: u64) -> Option<CorpusOutcome>;
// CorpusOutcome { adopt: bool, ... }   // Some(adopt=true) ⇒ Pass, Some(adopt=false) ⇒ Fail
```

and its call site (`verify_candidate_impl`, already live from A4/D5) does:

```rust
let corpus = advisory.and_then(single_primitive_optag).and_then(|op| corpus_verdict(op, out_dtype, seed));
if let Some(c) = &corpus { return classify_floor_verdict(None, None, Some(c)); } // corpus is authoritative
```

For a `Some` return to be **correct**, `corpus_verdict` must decide adopt/reject
*for the specific candidate under test*. It cannot, because:

1. **No candidate output crosses the seam.** `corpus_verdict` gets only
   `(op, dtype, seed)`. It never sees `cand_out.bytes`, so it cannot compare the
   candidate against the corpus's `expected`.
2. **`seed` is a random-probe seed, not a corpus selector.** The candidate was
   run once on `probe = probe_from_operands(operands, seed)` with
   `seed = 0x5EED_C0DE_1234_5678 ^ kernel_revision_hash`. That probe is a
   pseudo-random input tuple — **disjoint** from the corpus's fixed hand-drafted
   inputs (e.g. `a=-0.0, b=0.0`). So even the candidate output already computed
   at the call site (`cand_out.bytes`) corresponds to the *wrong* inputs to
   compare against a corpus `expected`.

Given (1)+(2), any non-`None` return would be an **adopt/reject on mere
`(op, dtype)` coverage** — `classify_floor_verdict` would Pass or Fail a
candidate the corpus never actually checked. That is precisely the "forced wrong
reading" the A4b guard forbids. The honest result: `corpus_verdict` returns
`None`; recipe-realize stays the interim authority.

This is a genuine gap in the **original A4/D5 design**, not new to A4b: keeping
`corpus_verdict` dormant "because the corpus is absent" masked that its signature
was never sufficient to be authoritative. The corpus's arrival exposes it.

## What activation needs (future increment — out of A4b scope)

To make the frozen corpus authoritative, the candidate must be evaluated on the
**corpus's own input vectors** and compared byte-exact. Two shapes work:

- **A. Corpus-driven probe.** When a covering cell exists for the claimed
  `(op, dtype)`, build the verify probe *from* the corpus inputs (instead of the
  seeded random probe), invoke the candidate on it, and compare `cand_out` to the
  corpus `expected` byte-exact (`ulp_bound == 0`). Verdict: adopt iff every
  covered cell matches; reject naming the first mismatching `tcId`. This reuses
  the existing invoke path; `corpus_verdict` becomes a helper that yields the
  probe + expected bytes, and the *comparison* moves to the call site (which has
  `cand_out`).
- **B. Widen the seam.** `corpus_verdict(op, dtype, invoke_fn)` where `invoke_fn`
  runs the candidate on arbitrary inputs; `corpus_verdict` drives each corpus
  vector and returns the aggregate `CorpusOutcome`. Heavier, but keeps the
  authority inside `corpus_verdict`.

Either way, mind:

- **Endianness.** Corpus `bits` are big-endian value bytes (MSB-first). Fuel's
  tensor storage is little-endian. The reader stores bytes verbatim; the
  consumer must swap before comparing (or compare on values with a raw-bit tie
  rule, per the corpus `harness_rule`: `0.0` vs `-0.0` must compare by bits).
- **Op/dtype key mapping.** `kiss_corpus` keys are strings (`"add"`, `"f32"`).
  The seam keys on `OpTag`/`DType`; add an `OpTag → &str` / `DType → &str`
  mapping (or reuse `fuel-kiss-ref-backend::mapping`) at wire time.
- **Coverage scope.** Only `(add, f32)` is populated today. Multi-node regions
  have no single-op cell (`single_primitive_optag → None`), so they correctly
  stay `None`.

## What A4b actually shipped

- Vendored `op_manifest.json` + `ops-arith.json` (KISS `c9153b2`) under
  `fuel-dispatch/fixtures/kiss-corpus/` with a `PROVENANCE.md`.
- `fuel-dispatch/src/kiss_corpus.rs`: a never-panic reader
  (`load_vendored_corpus`, `Corpus::{covers, cells, covered_cells, declares_op}`)
  with `#[cfg(feature = "jit")]` gating and unit tests over the `(add, f32)` cell.
- `corpus_verdict` unchanged (returns `None`); its doc updated to point here.
