# Port: Quantized Whisper (GGUF)

## Eager source

- `fuel-transformers/src/models/audio/whisper/quantized_model.rs` (411 LOC)
  — GGUF-quantized variant of Whisper. Same architecture as
    plain Whisper, but linear layers swap for Q-matmul against
    GGUF Q4_0 / Q4_K_M / Q5_0 / Q8_0 weights.

## Lazy module name

Extend `fuel-core/src/lazy_whisper.rs` with a sibling
`QuantizedWhisper` (or add a new module
`lazy_quantized_whisper.rs` — match whatever pattern
`lazy_quantized_smollm3` and other quantized siblings use).

## Architecture summary

Whisper is encoder-decoder for ASR. The quantized variant:
- Replaces `Linear` weights with GGUF-quantized blocks loaded via
  the existing GGUF loader.
- Q-matmul ops route through the shipped `Nf4Matmul` + GGUF
  fused-op surface.
- Layer norms, conv front-end, softmax, RoPE — unchanged (no
  quant on those).
- Encoder and decoder share the same Q-matmul substitution
  pattern.

## Primitives needed

- All shipped:
  - GGUF Q-matmul fused-op surface.
  - Whisper architecture (`lazy_whisper`).
  - GGUF loader pattern (used by other quantized ports).

## Reusable modules

- `lazy_whisper` — the plain Whisper model. The quantized variant
  parallels it shape-for-shape, just with Q-weights.
- Existing `lazy_quantized_*` ports (smollm3, mistral, etc.) — for
  the GGUF loader + Q-matmul substitution pattern.

## Open questions

- Which quantization types are supported? Q4_0 + Q4_K_M + Q5_0 +
  Q8_0 are the standard set; verify against the eager file's
  GGUF loader.
- Cross-attention layers in the decoder — are those quantized too,
  or only self-attention?
- Conv front-end (the two strided conv1ds in the Whisper encoder) —
  quantized or kept f32?

## Splits

Single session — mostly mechanical substitution of
`linear(...)` → `qmatmul(...)` against the existing
`lazy_whisper` forward path.

## Test strategy

- Tiny config matching the plain `lazy_whisper` smoke test, with
  Q-weights synthesized from f32 → Q4_0 round-trip.
- Output finite, shape matches plain Whisper.
- Optional integration: load a real `ggml-tiny.q4_0.bin` and
  verify a short clip transcribes to the same tokens as a
  reference run (mark `#[ignore]` so it doesn't run in CI).

## References

- Eager source: `fuel-transformers/src/models/audio/whisper/quantized_model.rs`
- Whisper paper: <https://arxiv.org/abs/2212.04356>
- whisper.cpp (GGUF format reference):
  <https://github.com/ggerganov/whisper.cpp>
- Already-shipped: `lazy_whisper`, `Nf4Matmul`, GGUF loader siblings.
