# gguf_corpus_digests.json — provenance

- **What:** for each of 30 `.gguf` files in the local corpus `C:\Models\gguf-corpus` (llama.cpp
  `ggml-vocab-*` files, `SmolLM2-135M-Instruct` quantization samples, tinyllamas), the FNV-1a 64
  hash (hex) of the `Debug` text of a digest of the parsed `Content`: magic, the whole metadata
  map, the tensor-info table, the unsupported-tensor table and the tensor-data offset.
- **Produced by:** fuel's OLD reader, `fuel_formats::gguf::Content::read` (the byte-stream parser
  deleted in the same change), run on 2026-10-24 at `origin/main@8e38ea8a` by the test that
  preceded this one, in which `Content::open` was compared with it on all 31 corpus files:
  30 identical, 1 refused by `open` (GGUF v1, `legacy/tinyllamas-stories-260k-f32.gguf`, which
  `mlmf-gguf` rejects on purpose). Both readers' digests were equal for every file listed here;
  the hash recorded is the one both produced.
- **Not here:** the corpus files themselves (1.2 GB, third-party models; never committed). Their
  names and sha256 are in the corpus's own `MANIFEST.json`.
- **Limits:** a hash proves equality with what the old reader produced on these files, not that
  the old reader was right; the independent check on correctness is MLMF's own tests. Four known
  parity differences between the readers (duplicate keys keep MLMF's first-wins rather than the
  old last-wins; a trailing NUL in a string is kept; a bool byte 2..=255 is accepted; a non-UTF-8
  key or tensor name is refused) do not occur on any of these files.
