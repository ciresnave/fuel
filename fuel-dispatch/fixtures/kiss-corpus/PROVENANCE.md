# Vendored KISS conformance corpus — provenance

**Source repo:** KISS (ThinkersJournal/KISS, private) — `conformance/corpus/`
**Pinned commit:** **per file** -- see the table. Every file is at KISS `5b28bf51`
(sk5 / `contract_version` 3, conformance 0.2.1) or has the SAME blob there:
`ops-arith`, `ops-minmax-ordinary` and `ops-minmax-signed-zero` were vendored
at `f4952b4c` (2026-09-02) and are byte-identical at `5b28bf51`. `dtype_manifest`
and `structure_key_vectors` were vendored 2026-10-11; `op_manifest` and the three
NaN files 2026-10-14. Three files moved to KISS `0af5d3b6` on 2026-10-14 (second pass):
`ops-transcendental-nan` (gained a `harness_rule`; same 12 vectors), and the new
`ops-minmax-nan` and `ops-rounding`.
**Vendored by:** fuel GAP-236 Step 0 (re-vendor + loader extension, 2026-09-02);
the sk5 deriver binding (2026-10-11) re-vendored two files; the NaN-vector
vendoring (2026-10-14) added `ops-narrow-move-nan`, `ops-narrow-select-nan`,
`ops-transcendental-nan` and moved `op_manifest` to `5b28bf51`; the evaluator pass
(2026-10-14) re-vendored `ops-transcendental-nan` and added `ops-minmax-nan` and
`ops-rounding` at `0af5d3b6`.
**Supersedes:** the `c9153b2` pin of 2026-07-23 (2 files)

## Files (copied verbatim — DO NOT EDIT)

| file | pinned at | KISS blob (`git ls-tree <pin> conformance/corpus/`) | schema |
|------|-----------|-------------------------------------------------------|--------|
| `op_manifest.json` | **`5b28bf51`** | `f656b625ab8e5371c7a4a73df92863d193c31b74` | `kiss-op-manifest-v1` |
| `ops-arith.json` | `f4952b4c` (same blob at `5b28bf51`) | `aee2de79a709e722f3c439047bf90c638aec9b24` | `kiss-oracle-vectors-v1.json` |
| `ops-minmax-ordinary.json` | `f4952b4c` (same blob at `5b28bf51`) | `7bd6cc394bd247f9ca34edfc957d70a37432af97` | `kiss-oracle-vectors-v1.json` |
| `ops-minmax-signed-zero.json` | `f4952b4c` (same blob at `5b28bf51`) | `d06bdb85c8e626750318b61585b4af3ea5c8d4b4` | `kiss-oracle-vectors-v1.json` |
| `dtype_manifest.json` | **`5b28bf51`** (same blob at `a33282c`) | `24ff38f068aa18be723c07d307c86ae572c1333f` | `kiss-dtype-manifest` |
| `structure_key_vectors.json` | **`5b28bf51`** | `e5eec85c82754f030f1ee2f0ab4bf7829fa7e5d2` | `kiss-structure-key-vectors` |
| `ops-narrow-move-nan.json` | **`5b28bf51`** | `c108226af556aa1c6a45e8e43b85ab29b96dec2d` | `kiss-oracle-vectors-v1.json` |
| `ops-narrow-select-nan.json` | **`5b28bf51`** | `2efbfd259c43936d8cf6dc24b1375f6c9b5ae81b` | `kiss-oracle-vectors-v1.json` |
| `ops-transcendental-nan.json` | **`ff9f52f5`** (blob first at `9ea8c05d`) | `d4f8b7486ca2b082252f4d305dff8edff2db5672` | `kiss-oracle-vectors-v1.json` |
| `ops-minmax-nan.json` | **`0af5d3b6`** | `576d69ab693f6acb835c1c91307dba26bb0390d3` | `kiss-oracle-vectors-v1.json` |
| `ops-rounding.json` | **`0af5d3b6`** | `f8f84610bc59bffe4e77c1ea2546482668fe5d77` | `kiss-oracle-vectors-v1.json` |

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

**2026-10-14 -- the paragraph that stood here is superseded.** It said `op_manifest.json`
was left at `f4952b4c` on purpose and that the NaN files were "not vendored and not
read". Both are now done, each as its own change with its own gate: `op_manifest.json`
moved to `f656b625...` (106 -> 122 ops: `argmin` and the 15 complex ops `cadd`..`cmake`;
nothing removed; a new `coverage` object `{declared: 1, total: 122}`), and the three
oracle-vector NaN files were vendored and read (below). **Still NOT vendored, on
purpose:** `announce_vectors`, `contract_vectors`, `grammar_vectors`,
`opattrs_vectors`. They are Announce/Contract/OpAttrs vectors with their own schemas;
Fuel claims Classify + Ops evaluation only (`docs/outreach/kiss-conformance-and-divergences.md`),
so there is no reader for them and vendoring them would only make the directory look
current.

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

Vector counts: `ops-arith` 5, `ops-minmax-ordinary` 24, `ops-minmax-signed-zero` 48
(**77**, the 2026-09-02 set), plus `ops-narrow-move-nan` 30, `ops-narrow-select-nan` 40,
`ops-transcendental-nan` 12 (**82**, 2026-10-14), plus `ops-minmax-nan` 44 and
`ops-rounding` 72 (**116**, KISS `0af5d3b6`) -- **276 total** (13 transcendental, KISS#550 added tcId 13: f64 `log` of a signalling NaN), of which 263 are
`exact-byte` and the 13 transcendental ones are class `ULP` with a NaN expected,
compared by NaN-ness and quietness (KISS-CONFORM-6.8-0010), never by ULP distance.

`ops-minmax-signed-zero.json` is about **signed-zero tie-breaking**, not NaN.
`+0` and `-0` compare EQUAL, so on a tie the result is decided by operand order
rather than by value. Its own `harness_rule` states the consequence: *"a
value-compare of `0.0 == -0.0` passes vacuously; the expected values are BIT
PATTERNS."* Any implementation compared by value passes these without testing
anything.

## 2026-10-14 update: the NaN axis is covered for the narrow dtypes AND f32/f64

Measured over all vendored minmax vectors (`max_prop` 38, `min_prop` 38, `fmax_ieee` 40,
`fmin_ieee` 40) grouped into `(dtype, input-bytes)` cells, at KISS `0af5d3b6`:

```
distinct cells                                   42   (was 18 at f4952b4c, 30 at 5b28bf51)
cells where MAX disagrees with MIN               control, > 0
cells where PROP disagrees with IEEE             18   (was 0, then 8)
  of which carry a NaN operand                   18   (all of them)
  bf16 f16 f8e5m2 f8e4m3fn                       2 each (ops-narrow-select-nan)
  f32 f64                                        5 each (ops-minmax-nan)
```

Every discriminating cell carries a NaN operand, and a moved sNaN keeps its payload
(`ops-narrow-move-nan`, KISS-OPS-6.16-0009). **The blind spot the next section describes is
CLOSED on the corpus side**: an `fmaxf`/`fmax` mis-lifted as a NaN-propagating `Max` on
`f32` or `f64` now fails five cells per dtype.
`kiss_corpus::tests::the_corpus_discriminates_prop_from_ieee_minmax_for_f32_and_f64` pins
that, replacing the tripwire that went red on vendoring `ops-minmax-nan`.

**These vectors ARE now run against Fuel**, through the lazy `Tensor` on CPU:
`fuel-tensor/tests/kiss_nan_conformance.rs` evaluates 199 vectors (all of the NaN files, plus
`ops-minmax-nan` and `ops-rounding`): **163 pass, 0 fail, 36 not exercised** (on x86 Windows; the macOS leg is CI). The 36 are
the two 8-bit float dtypes (34) and `copysign` (2), which Fuel has no CPU `Tensor` path or
op for; they are unmeasured, not passes (registry rows GAP-352 and GAP-353).
`ops-rounding` is what `trunc` never had: Fuel's own expansion passes all 18 `f32`
`trunc` cells (signed zero, halves and ties, 2^23, max finite, +-inf, subnormals).

## (SUPERSEDED 2026-10-14 for the narrow dtypes; text below is the 2026-09-02 measurement at `f4952b4c`) WHAT THIS CORPUS DOES NOT COVER — measured, not assumed

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

Evaluator: `fuel-tensor/tests/kiss_nan_conformance.rs` (it reads these same files by
relative path, because `fuel-dispatch` cannot dev-depend on `fuel-tensor`). A NaN-expected
`ULP` cell is compared per KISS-CONFORM-6.8-0010; anything else bit-exact.

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
