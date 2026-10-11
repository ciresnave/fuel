# Vendored KISS conformance corpus — provenance

**Source repo:** KISS (ThinkersJournal/KISS, private) — `conformance/corpus/`
**Pinned commit:** **per file** — see the table. Two files are pinned at KISS
`5b28bf51` (sk5 / `contract_version` 3, conformance 0.2.1, vendored 2026-10-11);
the other four are still at `f4952b4c` (2026-09-02). Three of those four have
the same blob at `5b28bf51`; `op_manifest.json` does not, and was deliberately
left at `f4952b4c` (see below).
**Vendored by:** fuel GAP-236 Step 0 (re-vendor + loader extension, 2026-09-02);
the sk5 deriver binding (2026-10-11) re-vendored the two files marked below.
**Supersedes:** the `c9153b2` pin of 2026-07-23 (2 files)

## Files (copied verbatim — DO NOT EDIT)

| file | pinned at | KISS blob (`git ls-tree <pin> conformance/corpus/`) | schema |
|------|-----------|-------------------------------------------------------|--------|
| `op_manifest.json` | `f4952b4c` | `9c7176ed4c16e34844f01abd826b0c9a74a6c461` | `kiss-op-manifest-v1` |
| `ops-arith.json` | `f4952b4c` (same blob at `5b28bf51`) | `aee2de79a709e722f3c439047bf90c638aec9b24` | `kiss-oracle-vectors-v1.json` |
| `ops-minmax-ordinary.json` | `f4952b4c` (same blob at `5b28bf51`) | `7bd6cc394bd247f9ca34edfc957d70a37432af97` | `kiss-oracle-vectors-v1.json` |
| `ops-minmax-signed-zero.json` | `f4952b4c` (same blob at `5b28bf51`) | `d06bdb85c8e626750318b61585b4af3ea5c8d4b4` | `kiss-oracle-vectors-v1.json` |
| `dtype_manifest.json` | **`5b28bf51`** (same blob at `a33282c`) | `24ff38f068aa18be723c07d307c86ae572c1333f` | `kiss-dtype-manifest` |
| `structure_key_vectors.json` | **`5b28bf51`** | `e5eec85c82754f030f1ee2f0ab4bf7829fa7e5d2` | `kiss-structure-key-vectors` |

Blob hashes were verified by computing `git hash-object --no-filters` on each
vendored file and matching the value `git rev-parse <pin>:conformance/corpus/<file>`
reports in the KISS checkout (positive control: the unchanged `ops-arith.json`
hashes to the same blob at both pins). For the two `5b28bf51` files the bytes were
taken with `git show 5b28bf51:<path>` (a commit object, never the working tree).
The `f4952b4c` files were read through the GitHub API at the pinned SHA rather
than from the local KISS checkout, whose `origin/main` was stale (`9d36c0a` vs
the true tip `f4952b4c`) — and fetching in a shared checkout moves refs under
other lanes. The
files were read through the GitHub API at the pinned SHA rather than from the
local KISS checkout, whose `origin/main` was stale (`9d36c0a` vs the true tip
`f4952b4c`) — and fetching in a shared checkout moves refs under other lanes.

## The sk5 re-vendor (2026-10-11): sk4 -> sk5, `contract_version` 3, at `5b28bf51`

KISS `a33282c` ("derive a 7-class cell_class, drop the 24 op-family codes (#263
ruling D) + dequant_form + extent-1 mask + math_fidelity (#535)") is one schema
event on two axes. **The pin is its follow-up `5b28bf51` (#536)**, not
`a33282c`: at `a33282c` seven reduction vectors still spelled the keepdim
OUTPUT broadcast mask `00` although KISS's own `derive_bcast_mask_in_frame`
derives the bit (`[4,1]` in a `[4,8]` frame is `0b10`); Fuel reported that, and
#536 regenerated them as `02`/`03`/`01`/`0a` (plus the two decline vectors that
embed those sub-keys). `a33282c` is the commit that CHANGED the schema; it is
not a vendorable corpus. What the event changes in the two files re-vendored here:

- `structure_key_vectors.json`: `token_prefix` `sk4` -> `sk5`,
  `structure_key_schema_version` 4 -> 5; field 1 is the derived `cell_class`
  (`ctr ndx rdc ew1 ew2 ew3 oth`), not one of 24 op-family codes; positives
  21 -> 26, declines 17 -> 21, `usable_count` 22 -> 21 (`f8e6m2` moved from
  usable to **reserved**, KISS #517); a new `target_match_vectors` axis (8
  rows, §6.8-0002) that the deriver does not exercise.
- `dtype_manifest.json`: `f8e6m2` is now in the `reserved` column.

⚠️ **`source_commit` inside `structure_key_vectors.json` is still `19c3ad7`** —
KISS's own hand-written stamp (KISS #218), unchanged by this event. It is NOT
the pin of this vendoring; the table above is. The leg's own provenance test
asserts both so a swap of either cannot pass quietly.

**Left at `f4952b4c` on purpose:** `op_manifest.json` differs at `a33282c`
(blob `f656b625…`), but nothing in the structure-key deriver reads it; moving it
would change `kiss_corpus.rs`'s inputs, which is a different change with a
different gate. Likewise the six NEW corpus files at `a33282c`
(`announce_vectors`, `contract_vectors`, `grammar_vectors`, `opattrs_vectors`,
`ops-narrow-*-nan`, `ops-transcendental-nan`) are **not** vendored here and
have **not been read** for this change: whether they close the NaN-vector ask in
the "does not cover" section below is unmeasured, and adopting them is its own
change with its own gate.

## Drift since the `c9153b2` pin — smaller than "six weeks stale" suggests

- `op_manifest.json` — **byte-identical.** Same blob as the 2026-07-23 pin.
- `ops-arith.json` — changed, and the **entire** diff is one line:
  `spec_clause` renumbered `KISS-CONFORM-6.4-0002` -> `KISS-CONFORM-6.5-0008`.
  **The five vectors are unchanged.** So the previously vendored oracle data was
  never wrong; only its citation was.

## What this corpus is

`kiss-oracle-vectors-v1` test vectors: each is a per-`(op, dtype, input-tuple)`
**exact-byte reference** — fixed input bit patterns and their single correct
output bit pattern (`class: "exact-byte"`, `ulp_bound: 0`), MSB-first hex. An
**oracle** (reference outputs for fixed inputs), NOT an `(op, dtype) ->
adopt/reject` verdict table.

Vector counts at this pin: `ops-arith` 5, `ops-minmax-ordinary` 24,
`ops-minmax-signed-zero` 48 — **77 total**.

`ops-minmax-signed-zero.json` is about **signed-zero tie-breaking**, not NaN.
`+0` and `-0` compare EQUAL, so on a tie the result is decided by operand order
rather than by value. Its own `harness_rule` states the consequence: *"a
value-compare of `0.0 == -0.0` passes vacuously; the expected values are BIT
PATTERNS."* Any implementation compared by value passes these without testing
anything.

## ⚠️ WHAT THIS CORPUS DOES NOT COVER — measured, not assumed

**It cannot distinguish NaN-propagating minmax (`max_prop`/`min_prop`) from IEEE
minmax (`fmax_ieee`/`fmin_ieee`).**

Measured over both minmax files at this pin:

```
distinct (dtype, input-pair) cells        18
cells carrying all four ops               18
cells where MAX disagrees with MIN         6
cells where PROP disagrees with IEEE       0
```

This is structural rather than accidental, and the corpus says so itself: KISS
§6.6 signed-zero equality makes both comparisons true on a ±0 tie, so operand
`a` wins every tie **in all four ops**. NaN is the axis on which the two
families differ — and **the entire corpus contains no NaN input at all** (77
vectors, 0 with a NaN operand; scanner positive-controlled against qNaN f32,
sNaN f32 `7F801234`, qNaN f64 and qNaN bf16, so the zero is real absence rather
than a broken query).

**Consequence, stated so a reader does not over-credit this vendoring:** these
vectors buy signed-zero tie-breaking and max-vs-min coverage that Fuel had none
of. They do **not** close the defect class GAP-236 is about — an `fmaxf`
mis-lifted as a NaN-propagating `Max` passes every one of them. A NaN-vector ask
is open with KISS: `(qNaN, finite)` and `(finite, qNaN)` per op per dtype, where
`max_prop` must propagate and `fmax_ieee` must return the finite operand.

`kiss_corpus::tests::the_corpus_cannot_yet_discriminate_prop_from_ieee_minmax`
pins that measurement. It **passes today and is expected to go RED** when the
NaN vectors land — that red is the signal to re-measure and close the row, and
it exists because a prose note would have gone silently stale instead.

## How Fuel reads it

Reader: `fuel-dispatch/src/kiss_corpus.rs`. Every vector file goes through one
`parse_vector_file`, so a second copy cannot drift from the first.

**Adding a file here is not enough to use it.** The reader takes files by name
through `include_str!`; a JSON dropped into this directory and not added to the
loader sits unread while the directory looks current.
`kiss_corpus::tests::the_minmax_vectors_are_loaded_not_merely_vendored` fails on
exactly that.

**Not** wired into `jit_ingest::corpus_verdict`: that seam's signature
(`(op, dtype, seed) -> Option<CorpusOutcome>`) carries no candidate output and its
`seed` selects a *random probe*, disjoint from these fixed corpus inputs — so it
cannot turn this oracle into a candidate verdict without re-running the candidate
on the corpus inputs. See `docs/design-notes/2026-07-23-kiss-corpus-verdict-seam-mismatch.md`.

## Re-vendoring

Copy the current `conformance/corpus/*.json` from KISS at a named commit, update
the pinned commit + blob hashes + date above, add any NEW file to the
`include_str!` set in `kiss_corpus.rs` (not just to this directory), and re-run
`cargo test -p fuel-dispatch --features jit --lib kiss_corpus`.
