# Port: Quantized SmolLM3 (GGUF)

## Eager source

- `fuel-transformers/src/models/llm/smol/quantized_smollm3.rs` (578 LOC)
  — GGUF-quantized variant of SmolLM3. Same architecture as plain
    SmolLM3 (`lazy_smollm3` is already shipped), but linear layers
    swap for Q-matmul against GGUF weights.

## Lazy module name

`fuel-core/src/lazy_quantized_smollm3.rs` (new file). Mirrors the
quantized-sibling pattern used by other quantized ports.

## Architecture summary

SmolLM3 is a small Hugging Face decoder-only LM with grouped-query
attention, RMSNorm, and RoPE. The quantized variant:
- Replaces `Linear` weights with GGUF Q-matmul (Q4_0 / Q4_K_M /
  Q5_0 / Q8_0).
- Layer norms, embedding, lm_head — unchanged.
- Forward path otherwise identical to `lazy_smollm3`.

## Primitives needed

- All shipped:
  - GGUF Q-matmul fused-op surface.
  - SmolLM3 architecture (`lazy_smollm3`).
  - GGUF loader pattern.

## Reusable modules

- `lazy_smollm3` — plain SmolLM3 forward.
- Other shipped `lazy_quantized_*` ports — pattern reference.

## Open questions

- Are embedding weights quantized in the GGUF file, or stored as
  f32? Standard GGUF for LLMs stores embedding as f16/f32 — verify
  against the eager loader.
- lm_head — typically tied to embedding; same precision.
- KV cache dtype — keep at f32 / f16 regardless of weight quant.

## Splits

Single session — mechanical substitution against `lazy_smollm3`.

## Test strategy

- Tiny config matching the plain `lazy_smollm3` test, with
  Q-weights synthesized from f32 → Q4_0 round-trip.
- Output finite, shape `(B, T, vocab)` correct.
- Optional `#[ignore]` integration test against a real
  `smollm3.q4_0.gguf`.

## References

- Eager source: `fuel-transformers/src/models/llm/smol/quantized_smollm3.rs`
- SmolLM blog: <https://huggingface.co/blog/smollm>
- GGUF format: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
- Already-shipped: `lazy_smollm3`, sibling quantized ports.
