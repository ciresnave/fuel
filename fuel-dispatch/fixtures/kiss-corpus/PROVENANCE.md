# Vendored KISS conformance corpus — provenance

**Source repo:** KISS (ThinkersJournal/KISS, private) — `conformance/corpus/`
**Pinned commit:** `c9153b2` (KISS `main`)
**Vendored on:** 2026-07-23
**Vendored by:** fuel task A4b (kiss-ref verdict integration, branch `feat/kiss-ref-verdict-integration`)

## Files (copied verbatim — DO NOT EDIT)

| file | KISS blob (`git ls-tree c9153b2 conformance/corpus/`) | schema |
|------|-------------------------------------------------------|--------|
| `op_manifest.json` | `9c7176ed4c16e34844f01abd826b0c9a74a6c461` | `kiss-op-manifest-v1` |
| `ops-arith.json`   | `8bab163ed2acc4fb88cc200e4493759758967f00` | `kiss-oracle-vectors-v1.json` |

These are byte-for-byte copies of the two files present under `conformance/corpus/`
at KISS `c9153b2`. The KISS working tree at vendoring time matched `c9153b2` for
both files (`git diff c9153b2 -- …` empty). A later KISS commit adds
`ops-minmax-signed-zero.json`; that file is NOT part of the `c9153b2` pin and is
NOT vendored here.

## What this corpus is

`ops-arith.json` is a set of `kiss-oracle-vectors-v1` test vectors: each vector is
a per-`(op, dtype, input-vector)` **exact-byte reference** — fixed input bit
patterns and their single correct output bit pattern (`class: "exact-byte"`,
`ulp_bound: 0`), MSB-first hex. It is an **oracle** (reference outputs for fixed
inputs), NOT an `(op, dtype) → adopt/reject` verdict table.

## How Fuel reads it

Reader: `fuel-dispatch/src/kiss_corpus.rs` (parses these files via `include_str!`).

**Not** wired into `jit_ingest::corpus_verdict`: that seam's signature
(`(op, dtype, seed) -> Option<CorpusOutcome>`) carries no candidate output and its
`seed` selects a *random probe*, disjoint from these fixed corpus inputs — so it
cannot turn this oracle into a candidate verdict without re-running the candidate
on the corpus inputs. See `docs/design-notes/2026-07-23-kiss-corpus-verdict-seam-mismatch.md`.

## Re-vendoring

Copy the current `conformance/corpus/{op_manifest.json,ops-arith.json}` from the
KISS checkout, update the pinned commit + blob hashes + date above, and re-run
`cargo test -p fuel-dispatch --features jit --lib kiss_corpus`.
