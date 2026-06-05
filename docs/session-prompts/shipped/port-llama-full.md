# Port: Llama full (LlamaConfig + Llama3 RoPE scaling + from_gguf hook)

## Eager source

- `fuel-transformers/src/models/llm/llama.rs` (673 LOC)
  — Full LLaMA-1/2/3 decoder with `LlamaConfig` (HF `config.json`
    deserializer), `Llama3RopeConfig` (per-band frequency scaling),
    `LlamaEosToks` (single vs multiple EOS), KV-cache helpers, and
    the `Cache::new` / `Llama::load` weight-load hooks.

## Lazy module name

`fuel-core/src/lazy_llama_full.rs` (new file). The existing
`lazy.rs` already carries an inline `LlamaModel` shape used by
LLaVA composition and the anchor-oracle tests — keep that as-is;
the new module adds the public `LlamaConfig` / `Llama3RopeConfig`
HF-deserializer + the from_hub loader on top.

## Architecture summary

LLaMA decoder-only transformer with RMSNorm, RoPE, SwiGLU, and
GQA. Architecture matches Mistral (which `lazy_mistral` already
ports) but with three LLaMA-specific pieces missing from
`lazy_mistral`:

1. **Llama-3 RoPE per-band scaling** — splits frequencies into
   three regions (low / mid / high) based on wavelength relative to
   `original_max_position_embeddings`:
   - wavelen < high_freq_wavelen: unscaled
   - wavelen > low_freq_wavelen: scaled by `1/factor`
   - in between: smooth interpolation by `(orig_max / wavelen) / factor`
   Replaces the standard `theta^(-2i/d)` frequency table.
2. **`LlamaEosToks` enum** — Hub configs sometimes list multiple
   EOS tokens (e.g., LLaMA-3 has `[128001, 128008, 128009]`).
   Pure config data; no graph impact.
3. **`LlamaConfig::into_config` adapter** — maps the HF JSON shape
   to the internal config the model consumes.

## Primitives needed

- [ ] Llama-3 RoPE table builder. Three-band scaling formula.
      Lives in `lazy_llama_full::build_llama3_rope_tables(cfg, seq, head_dim)`.
      Returns `(cos, sin)` Arc<[f32]> for the seq * head_dim
      grid; the table is computed host-side and emitted as
      `const_f32_like` at forward time, same as the standard
      RoPE tables already used by lazy_mistral.

## Reusable modules

- `lazy_mistral` — Mistral is structurally LLaMA + sliding window.
  Use its forward path directly; just swap the RoPE tables to the
  Llama-3-scaled variant when `rope_scaling.rope_type == Llama3`.
- `lazy.rs` inline `LlamaModel` — load-bearing for LLaVA and
  anchor-oracle tests. Leave untouched; the new
  `lazy_llama_full::LlamaModel` is a separate public type that
  composes over `lazy_mistral`.
- `crate::safetensors::MmapedSafetensors` — for the HF safetensors
  loader.
- `lazy_resnet`'s `load_f32` / `load_transposed` helpers — pattern
  to copy for the safetensors loader.

## Open questions

- Should `lazy_llama_full::LlamaModel` re-export
  `lazy_mistral::MistralModel`'s forward signature directly, or
  wrap it in a `LlamaModel` newtype? Preference: newtype wrapper
  so the public surface looks like LLaMA, not Mistral.
- LLaMA-3.1 has further RoPE changes (extended context to 128k via
  `Llama3RopeConfig` factor=8). Verify the three-band formula
  matches HuggingFace transformers' Llama-3.1 implementation
  exactly. (Pull from
  `transformers/models/llama/modeling_llama.py` if any drift.)
- GGUF loader — what's the minimum surface? `from_gguf(path)` that
  produces a working `LlamaModel`? Check what `lazy_quantized_*`
  ports do for their GGUF loaders.

## Splits (if work needs to be broken into smaller prompts)

This port is bounded enough for one session. If it grows:

1. Sub-port 1: `LlamaConfig` + `Llama3RopeConfig` + `LlamaEosToks`
   types with HF JSON Deserialize + unit tests for the three-band
   RoPE math.
2. Sub-port 2: `LlamaWeights::load_from_mmapped` + `LlamaModel::from_hub`
   reusing `lazy_mistral::MistralModel` for forward.
3. Sub-port 3: GGUF loader (gated on existing GGUF loader pattern
   in `lazy_chatglm::ChatGlmConfig::codegeex4` or similar).

## Test strategy

- Tiny config: `vocab_size=32`, `hidden_size=16`, `intermediate_size=32`,
  `num_hidden_layers=2`, `num_attention_heads=4`,
  `num_key_value_heads=2`, `head_dim=4`, `rope_theta=10000`,
  `rope_scaling=None` (standard RoPE) — exercises the LLaMA-2
  path.
- Same config but with `rope_scaling=Some(Llama3RopeConfig {
  factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0,
  original_max_position_embeddings: 8192, rope_type: Llama3 })`
  — exercises the LLaMA-3.1 per-band path. Assert the RoPE table
  differs from the unscaled variant.
- `from_hf_json_str_parses_canonical_llama3_fields` — round-trip a
  real HuggingFace Llama-3.1-8B `config.json` (paste a snippet
  into the test).
- `forward_embeds_matches_forward_after_token_lookup` — this port
  reuses `lazy_mistral`'s forward, which already has the
  forward_embeds contract; just verify the wrapper preserves it.

## References

- Eager source: `fuel-transformers/src/models/llm/llama.rs`
- HuggingFace reference:
  `transformers/src/transformers/models/llama/modeling_llama.py`
  (look at `LlamaRotaryEmbedding` for the three-band scaling
  formula and the `factor` / `low_freq_factor` / `high_freq_factor`
  semantics).
- Already-shipped similar ports:
  - `lazy_mistral` (forward path — LLaMA-shape + sliding window)
  - `lazy_gemma3` (`forward_embeds` + dual RoPE pattern)
  - `lazy_llama2c` (HF safetensors `load_from_mmapped` + `from_hub`)
- LLaMA-3 RoPE explainer (Meta):
  <https://ai.meta.com/blog/meta-llama-3/> — section on "expanded
  context window" describes the scaling intuition.
