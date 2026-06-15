# fuel-quantized-lfm2

Fuel implementation of GGUF-quantized LFM2 (Liquid Foundation Model 2),
a hybrid attention + short-conv (LIV) architecture from LiquidAI.

## Running

```bash
$ cargo run --example quantized-lfm2 --release -- \
    --model path/to/LFM2-1.2B-Q4_K_M.gguf \
    --tokenizer path/to/tokenizer.json \
    --prompt "Tell me a story in 100 words." \
    --sample-len 200
```

Both `--model` and `--tokenizer` are required — the example does not
hard-code a HuggingFace repo for LFM2 (the LiquidAI releases use a
custom tokenizer that varies per model size).

## What's running under the hood

- `fuel::lazy_quantized_lfm2::QuantizedLFM2Model::from_gguf` parses the
  GGUF header, derives the per-layer block schedule from
  `lfm2.attention.head_count_kv` (non-zero entries = Attention, zero
  entries = ShortConv / LIV), keeps Q4_0 tensors quantized, and
  dequantizes other GGML dtypes (F16 / BF16) to F32. Q4_K_M tensors
  are rejected at load time — they currently need to be re-quantized
  to Q4_0 upstream until a native Q4_K_M matmul lands.
- The greedy/temperature/top-k/top-p sampling path mirrors the other
  `quantized-*` examples. `--temperature 0` selects argmax.

## Caveats

- **Prefill-only ShortConv state.** Single-step ShortConv requires a
  persistent `[B, hidden, l_cache]` cache that the v1 lazy port does
  not maintain — `forward(&[next_token], pos)` rebuilds the conv from
  zero state each call. Output quality may drift on long generations
  until the multi-output infrastructure lands (see
  `docs/session-prompts/shipped/multi-output-nodes-option-c.md`).
- **Q4_K_M.** Tensors stay Q4_0 in the lazy matmul; any non-Q4_0
  Linear tensor in the GGUF dequantizes to F32 at load time.

## Historical example output (eager v0)

The pre-lazy eager prototype produced output like:

```text
$ cargo run --example quantized-lfm2 --release -- --prompt "Tell me a story in 100 words."
A quiet town nestled between rolling hills, where every springtime arrives with laughter and blossoms.
Clara, the town's beloved baker, opens her shop at dawn — cinnamon swirling into warm air, fresh
pastries glowing on wooden racks. ...
```

The current lazy-graph path is correctness-tested; per-token throughput
will improve once ShortConv state caching lands.
