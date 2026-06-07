# fuel-metavoice (lazy v1)

MetaVoice-1B is a text-to-speech model trained on 100K hours of speech,
more details on the [model card](https://huggingface.co/metavoiceio/metavoice-1B-v0.1).

This binary is a lazy-port revival that wires
`fuel::lazy_metavoice` (the stage-2 multi-codebook transformer) to
`fuel::lazy_encodec` (the EnCodec decoder).

## Known gaps vs. the retired eager binary

- **Stage-1 GPT model is not lazy-ported.** The eager pipeline first
  ran a stage-1 GPT to produce a 2-codebook stream + a text-id
  adapter, then fed that into the stage-2 transformer. This binary
  drives the stage-2 transformer directly on the prompt tokens and
  emits its native `num_codebooks` (default 4) codes per position,
  feeding them straight into EnCodec. Quality is below the eager
  pipeline as a result.

- **BPE tokenizer is not lazy-ported.** The eager binary loaded a
  tiktoken-format BPE from `first_stage.meta.json`. We fall back to a
  byte-level encoding so the model can be exercised end-to-end.

- **Speaker embedding mel pipeline is not lazy-ported.** Pass
  `--speaker-encoder <path/to/spk_emb.safetensors>` to load a
  pre-baked `spk_emb` tensor (eager convention); otherwise we use a
  zero speaker vector.

## CLI

```text
--prompt           Text to synthesize.
--speaker-encoder  Optional safetensors with a `spk_emb` tensor.
--first-stage      EnCodec decoder weights (default: facebook/encodec_24khz).
--second-stage     MetaVoice stage-2 transformer weights (default: lmz/fuel-metavoice).
--output-wav       Output wav path (default: out.wav).
```

## Run an example

```bash
cargo run --example metavoice --release -- \
  --prompt "This is a demo of text to speech by MetaVoice-1B." \
  --output-wav out.wav
```
