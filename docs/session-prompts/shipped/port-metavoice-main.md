# Port: MetaVoice main LM

## Eager source

- `fuel-transformers/src/models/audio/metavoice.rs` (1072 LOC)
  — Main LM for the MetaVoice TTS model. The speaker encoder is
    already shipped as `lazy_metavoice_speaker_encoder`.

## Lazy module name

`fuel-core/src/lazy_metavoice.rs` (new file). The speaker encoder
already lives at `lazy_metavoice_speaker_encoder`.

## Architecture summary

MetaVoice is a text-to-speech (TTS) LM from MetaVoice-1B. The main
LM is a decoder-only transformer that predicts EnCodec audio
tokens conditioned on text + speaker embedding.

Components inside the 1072 LOC:
- TransformerLM forward (standard decoder-only with RMSNorm/RoPE).
- Speaker conditioning: speaker embedding from
  `lazy_metavoice_speaker_encoder` is concatenated or added at the
  embedding layer.
- Multi-codebook prediction head: predicts N parallel codebook
  streams (EnCodec) at each step.
- Sampling logic (top-k / top-p / temperature) — likely host-side
  scalar control.
- Text tokenizer integration.

## Primitives needed

- Standard decoder primitives (all shipped).
- Multi-codebook output head — `LazyTensor` reshape + per-codebook
  classifier.

## Reusable modules

- `lazy_metavoice_speaker_encoder` — speaker embedding upstream.
- `lazy.rs` or `lazy_mistral` for decoder forward pattern.
- `lazy_dac` / `lazy_encodec` — audio codec downstream (decode the
  predicted tokens back to PCM).

## Open questions

- Sampling — is it in this file, or expected to be in a separate
  sampling loop? Probably embedded; lift if needed.
- Codebook count and dim — config-driven.
- Cross-attention or pure concat-conditioning for speaker
  embedding?
- Does this depend on EnCodec being on the lazy graph? EnCodec
  decoder exists as `lazy_encodec`, so yes — but it's already
  shipped.

## Splits

Recommended split:

1. **Sub-port 1**: TransformerLM forward + RMSNorm/RoPE +
   multi-codebook head. Standalone test.
2. **Sub-port 2**: Speaker conditioning + sampling glue +
   end-to-end inference loop. Integration test (text → tokens →
   EnCodec → PCM shape check).

## Test strategy

- Tiny config: vocab=64, hidden=16, layers=2, heads=4, kv=2,
  codebooks=4. Forward 8 text tokens + 1 speaker embed → output
  `(B, num_codebooks, vocab)` finite logits.
- End-to-end: tiny config, generate 16 audio-token steps, decode
  through EnCodec, assert PCM shape `(1, samples)` finite.

## References

- Eager source: `fuel-transformers/src/models/audio/metavoice.rs`
- MetaVoice: <https://github.com/metavoiceio/metavoice-src>
- Already-shipped: `lazy_metavoice_speaker_encoder`,
  `lazy_encodec`, `lazy_dac`.
