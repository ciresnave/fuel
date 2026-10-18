# Third-party notices

Licence texts for third-party work this repository derives from, reproduced
verbatim from the upstream source at the stated ref. A file that derives from
one of these carries no SPDX identifier of ours; it is registered in
`fuel-ir/tests/spdx_header_ratchet.rs` (`INHERITED_WITH_NOTICE`), which fails
if the notice is removed.

## llama.cpp / ggml

- Origin: https://github.com/ggml-org/llama.cpp, `LICENSE` at commit `9d57ce456c`
  (the commit `GgmlDType` in `fuel-ir/src/quantized.rs` is pinned to).
- Used by: `fuel-quantized/src/iq4.rs` (IQ4_NL and IQ4_XS block layouts, codebook,
  dequantization arithmetic; verified against `ggml/src/ggml-common.h`,
  `ggml/src/ggml-quants.c`, `ggml/include/ggml.h` at that commit).

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
