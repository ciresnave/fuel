# File-level provenance (candle)

This repository began as a fork of [candle](https://github.com/huggingface/candle) (MIT OR Apache-2.0).
`FILES.tsv` classifies **every tracked file** against candle at the fork point. The licence texts and the
upstream contributors are in `THIRD-PARTY-NOTICES.md` (section "candle").

**Fork point.** The last pure-candle commit in this history is `b4b15281b` (2026-04-02); its tree hash
`3afd741e40196ab07f47dfc88eb10071f5ca36d1` is identical to upstream `huggingface/candle`
`097655a29e17ad4891bb1014ec7f3aee53e64e6d`, so this history's candle files ARE upstream's (compared by tree hash).

**Classes** (evidence, not a verdict on every idea in a file; the generator is `scripts/provenance.py`):

| class | meaning |
|---|---|
| `candle` | same path, or a git rename/copy, of a candle file, and at least 90% of its lines (over 25 characters) appear verbatim |
| `candle-modified` | lineage to a candle path (same path, or rename/copy at 30% or more similarity), or at least 30% of a source file's lines appear verbatim in candle |
| `ggml` | derived from llama.cpp / ggml (MIT), see the ledger |
| `not-traced` | no lineage, but at least 10% (or 20 lines) of a source file's lines also appear in candle, or it names candle in its text: **not yet judged by a person** |
| `data` | binary file with no match; provenance is its directory's `PROVENANCE.md` where present |
| `copy` | byte-identical packaged copy of a root file (`scripts/sync-packaged-notices.py`) |
| `own` | no lineage and no meaningful verbatim overlap with candle. **Not a proof of independent authorship**: it says only that this method found nothing |

**What this does NOT cover.** Verbatim-line matching cannot see a file re-derived from candle's structure line by
line in different words, nor work derived from a project other than candle (llama.cpp, MLX and bitsandbytes are in
the ledger where known; the `own` class is silent about any other source). Candle's licence files name no holder,
so nothing here names one.

**The gate.** `fuel-ir/tests/spdx_header_ratchet.rs` (`candle_provenance_table_covers_every_tracked_file`) fails when a
tracked file has no row in `FILES.tsv`, when a row names no tracked file, or when a row has an unknown class. A new
file therefore cannot land without someone classifying it. Regenerate with `python scripts/provenance.py generate`
(it reads only this repository's history).

## What we carry, and why the burden is small (2026-10-25)

CireSnave asked whether Fuel should remove the candle code it still carries so that candle's licences stop mattering. Short answer: **removal is not pursued** (CireSnave, 2026-10-25: "If the Candle license burden is small, don't worry about it").

**What we carry.** By the table below, of 624,176 lines of Rust: 13,830 are `candle` and 42,133 are `candle-modified` (together 9%, mostly `fuel-examples`, `fuel-metal-kernels`, `fuel-quantized`, `fuel-ir` and `fuel-cpu-backend`); 233,939 are `not-traced` (38%; 145,780 of them in `fuel-transformers`, whose lazy model files mirror candle's model directory but share almost no lines with it); 331,963 are `own`. The `not-traced` class is "no lineage, but a person has not judged it", not "derived". One of the 252 `not-traced` Rust files (`fuel-graph/src/lib.rs`, 17k lines) was read side by side with candle and judged independent; the others were not judged, because the study was stopped once the burden was found to be small. Four crates that nothing else in the workspace depends on (`fuel-examples`, `fuel-lazy-examples`, `fuel-posttrain`, `fuel-tensor-tools`) hold 49% of the `candle` and `candle-modified` lines (27,643 of 55,963) plus a further 12,013 `not-traced` lines; `fuel-transformers` is depended on by `fuel` and `fuel-inference`.

**Why the burden is satisfied.** Candle is dual-licensed MIT OR Apache-2.0 and Fuel uses it under MIT, which asks only that its notice and permission text travel with copies. Every publishable crate ships `LICENSE-MIT`, `LICENSE-APACHE` and `THIRD-PARTY-NOTICES.md` (candle's licence texts verbatim, the upstream commit, the fork point and the contributors), so the notice travels with the source a crates.io release contains. Candle's licence files name no holder, so none is named.

**How the ratchet keeps it so.** `fuel-ir/tests/spdx_header_ratchet.rs` fails if a publishable crate lacks a byte-identical copy of the three files, if a tracked file has no row in `FILES.tsv` (so a new file must be classified), or if a file that names its candle origin loses that statement; CI's "Packaged crates" job fails if a built tarball lacks any of the three files.

**Other upstreams Fuel carries, with their licences.** llama.cpp / ggml, MIT (the IQ, TQ and MX block formats and tables, and through candle the Q4_0, Q8_0 and k-quant formats); bitsandbytes, MIT (the NF4 decode); Apple MLX, MIT (`mlx_gemm.metal`, `mlx_sort.metal`); Jeffrey Quesnelle and Bowen Peng, MIT (a rotary-embedding section of `quantized.metal`); the `bs1770` loudness library by Ruud van Asseldonk, Apache-2.0 only (`fuel-examples/src/bs1770.rs`); the ONNX protobuf definition, Apache-2.0 only (`fuel-onnx/src/onnx.proto3`); and a 16-row excerpt of the Qwen3-0.6B weights, Apache-2.0 (a test fixture). Model code throughout cites PyTorch and Hugging Face behaviour in comments (116 files mention PyTorch); this study found no PyTorch or Hugging Face file carried in the tree, but it did not look for one exhaustively. All of these but the two Apache-only items are in `THIRD-PARTY-NOTICES.md` or are the project's own code under its dual licence.

## Summary by directory

<!-- provenance-summary:begin -->
| directory | files | candle | candle-modified | not-traced | ggml | own | data | copy |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `(root)` | 24 | 4 | 4 |  |  | 16 |  |  |
| `.cargo` | 1 |  | 1 |  |  |  |  |  |
| `.github` | 7 | 1 | 2 |  |  | 4 |  |  |
| `docs` | 193 |  |  |  |  | 193 |  |  |
| `fuel` | 8 |  |  |  |  | 5 |  | 3 |
| `fuel-aocl-cpu-backend` | 7 |  |  |  |  | 4 |  | 3 |
| `fuel-backend-contract` | 10 |  |  | 4 |  | 3 |  | 3 |
| `fuel-conv` | 6 |  |  |  |  | 3 |  | 3 |
| `fuel-correctness-fixtures` | 7 |  |  |  |  | 4 |  | 3 |
| `fuel-cpu-backend` | 30 | 2 | 3 | 5 |  | 17 |  | 3 |
| `fuel-cpu-kernels` | 12 | 1 | 4 |  |  | 4 |  | 3 |
| `fuel-cuda-backend` | 56 |  | 3 | 11 |  | 39 |  | 3 |
| `fuel-datasets` | 13 |  | 7 |  |  | 3 |  | 3 |
| `fuel-dispatch` | 300 |  |  |  |  | 297 |  | 3 |
| `fuel-examples` | 250 | 56 | 162 | 21 |  | 8 |  | 3 |
| `fuel-formats` | 11 |  | 2 | 2 |  | 4 |  | 3 |
| `fuel-graph` | 38 |  |  | 3 |  | 32 |  | 3 |
| `fuel-hardware` | 10 |  | 1 |  |  | 6 |  | 3 |
| `fuel-inference` | 26 |  |  | 2 |  | 21 |  | 3 |
| `fuel-ir` | 62 |  | 6 | 8 |  | 45 |  | 3 |
| `fuel-kernel-seam` | 5 |  |  |  |  | 2 |  | 3 |
| `fuel-kernel-seam-announce` | 6 |  |  |  |  | 3 |  | 3 |
| `fuel-kernel-seam-types` | 8 |  |  |  |  | 5 |  | 3 |
| `fuel-kernels-source` | 185 |  |  |  |  | 182 |  | 3 |
| `fuel-kiss-ref-backend` | 8 |  |  |  |  | 5 |  | 3 |
| `fuel-kv-pool` | 8 |  |  |  |  | 5 |  | 3 |
| `fuel-lazy-examples` | 27 |  |  | 21 |  | 3 |  | 3 |
| `fuel-loaders` | 21 |  | 1 | 3 |  | 14 |  | 3 |
| `fuel-memory` | 8 |  |  |  |  | 5 |  | 3 |
| `fuel-metal-backend` | 10 | 1 | 2 |  |  | 4 |  | 3 |
| `fuel-metal-kernels` | 53 | 43 | 4 | 1 |  | 2 |  | 3 |
| `fuel-mkl-cpu-backend` | 7 |  |  |  |  | 4 |  | 3 |
| `fuel-model-core` | 8 |  |  |  |  | 5 |  | 3 |
| `fuel-model-fixture` | 2 |  | 1 |  |  | 1 |  |  |
| `fuel-model-llama` | 8 |  |  | 1 |  | 4 |  | 3 |
| `fuel-model-phi` | 5 |  |  | 1 |  | 1 |  | 3 |
| `fuel-model-umbrella-fixture` | 2 |  |  |  |  | 2 |  |  |
| `fuel-nn` | 27 |  | 1 | 11 |  | 12 |  | 3 |
| `fuel-onnx` | 13 | 1 | 3 | 2 |  | 4 |  | 3 |
| `fuel-parallel` | 13 |  |  |  |  | 10 |  | 3 |
| `fuel-posttrain` | 6 |  |  | 2 |  | 1 |  | 3 |
| `fuel-quantized` | 16 |  | 4 | 1 | 4 | 4 |  | 3 |
| `fuel-tensor` | 65 | 1 | 2 | 4 |  | 53 | 2 | 3 |
| `fuel-tensor-tools` | 5 |  | 1 | 1 |  |  |  | 3 |
| `fuel-test-support` | 11 |  |  |  |  | 8 |  | 3 |
| `fuel-training` | 9 |  |  |  |  | 6 |  | 3 |
| `fuel-transformers` | 194 | 2 | 4 | 153 |  | 29 | 3 | 3 |
| `fuel-vulkan-backend` | 26 |  |  |  |  | 23 |  | 3 |
| `fuel-vulkan-kernels` | 185 |  |  |  |  | 2 | 180 | 3 |
| `scripts` | 75 |  |  | 1 |  | 74 |  |  |
| **all** | **2087** | **112** | **218** | **258** | **4** | **1181** | **185** | **129** |
<!-- provenance-summary:end -->
