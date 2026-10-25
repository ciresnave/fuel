# Third-party notices

Licence texts for third-party work this repository derives from, reproduced
verbatim from the upstream source at the stated ref. A file that derives from
one of these carries no SPDX identifier of ours; it is registered in
`fuel-ir/tests/spdx_header_ratchet.rs` (`INHERITED_WITH_NOTICE`), which fails
if the notice is removed.

## llama.cpp / ggml

- Origin: https://github.com/ggml-org/llama.cpp, `LICENSE` at commit `9d57ce456c`
  (the commit `GgmlDType` in `fuel-ir/src/quantized.rs` is pinned to).
- Also reached through candle (next section): the Q4_0 / Q8_0 / k-quant block layouts and quantization
  algorithms in `fuel-quantized/src/k_quants.rs`, `utils.rs`, `avx.rs`, `neon.rs`. Compared line by line only for
  `BlockQ4_0::from_float` (the same steps as `quantize_row_q4_0_ref` in `ggml/src/ggml-quants.c` at
  `9d57ce456c`); the rest rests on candle's own citations of llama.cpp commits in those files' comments.
- Used by: `fuel-quantized/src/iq4.rs` (IQ4_NL and IQ4_XS block layouts, codebook,
  dequantization arithmetic), `fuel-quantized/src/iq_grid_types.rs` (IQ2_XXS,
  IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M block layouts and dequantization),
  `fuel-quantized/src/tq_mx_types.rs` (TQ1_0, TQ2_0, MXFP4, NVFP4 block layouts, codebook
  and scale decoding) and `fuel-quantized/src/iq_grids.rs` (the IQ lookup tables, generated
  from `ggml/src/ggml-common.h` by `scripts/gen_iq_grids.py`); all verified against
  `ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c`, `ggml/src/ggml-impl.h`,
  `ggml/include/ggml.h` at that commit, and the IQ dequantizers additionally byte-compared with llama.cpp's own
  compiled C (see GAP-211 in `docs/gaps.md`).

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## candle (Hugging Face)

- Origin: https://github.com/huggingface/candle, dual-licensed MIT OR Apache-2.0. Fuel is a fork of it (this
  repository's history still carries candle's commits). Its `LICENSE-MIT` is byte-identical to this repository's
  root `LICENSE-MIT`, and its `LICENSE-APACHE` is the root `LICENSE-APACHE` apart from one leading blank line
  (both compared at upstream commit `549eacb011f25b2c6fc765d6c3dbd26e42f72ee6`, 2025-11-18, the upstream
  counterpart of `7b1d2a0a8`, the last candle-era commit here that touched these files). Neither upstream licence file names a holder.
- Used by (traced to `candle-core/src/quantized/` at that commit, then modified here):
  `fuel-quantized/src/k_quants.rs` (from `k_quants.rs`), `utils.rs` (from `utils.rs`), `avx.rs` (from `avx.rs`),
  `neon.rs` (from `neon.rs`), and, in part only, `cpu.rs` (the `QuantizedType` trait, from `mod.rs`).
  These files keep our SPDX line because the licence is the same one candle declares.
- Verified: the repository's own history blobs for those four files at `7b1d2a0a8` are byte-identical to
  upstream at `549eacb01`. Non-blank lines of the current file that also appear verbatim in the upstream file:
  `k_quants.rs` 2162 of 2257, `utils.rs` 428 of 518, `avx.rs` 613 of 752, `neon.rs` 536 of 697 (measured
  2026-10-25; `cpu.rs` 28 of 151 code lines over 25 characters against all of upstream `quantized/*.rs`).
- Not derived from candle by the same measure: `dequant.rs` (0 of 94 such lines verbatim; it dispatches to
  `GgmlType::to_float` and holds no block math), `awq_repack.rs` (written against AutoAWQ's layout and
  baracuda's kernel layout), `lib.rs`. Other candle-derived files elsewhere in this repository are NOT traced
  by this section.
- Upstream contributors to the traced files up to that commit, by commit count on those paths in candle's
  history: Laurent Mazare (83), ivarflakstad (10), Lukas Kreussel (8), Eric Buehler (6), Nicolas Patry (2),
  Jon Craton (2), zmlcc, zachcp, Kyle Birnbaum, Anubhab Bandyopadhyay (1 each).
- Licence texts: this repository's `LICENSE-MIT` and `LICENSE-APACHE`.

## bitsandbytes

- Origin: https://github.com/bitsandbytes-foundation/bitsandbytes, `LICENSE` at commit `833649043474794b8fe7a4136e0c40faf077b2e0`.
- Used by: `fuel-tensor/src/nf4.rs` (the double-quantization absmax decode `code[q] * nested_absmax[i / nested_blocksize] + offset` and the packed-nibble order, re-implemented from `bitsandbytes/functional.py` and `bitsandbytes/backends/default/ops.py`; no source copied).

```text
MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Qwen3-0.6B weights (test fixture)

- Origin: https://huggingface.co/Qwen/Qwen3-0.6B (Apache-2.0, Copyright 2024 Alibaba Cloud) and its bitsandbytes quantization https://huggingface.co/unsloth/Qwen3-0.6B-unsloth-bnb-4bit (apache-2.0).
- Used by: `fuel-tensor/tests/fixtures/bnb_double_quant/` (a 16-row excerpt of one weight matrix in quantized and original form, modified as listed in that directory's `PROVENANCE.md`). The licence text is `fuel-tensor/tests/fixtures/bnb_double_quant/LICENSE-Qwen3-0.6B-Apache-2.0.txt`.
