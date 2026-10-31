# qwen3_5_0_8b_forward_reference.safetensors

Intermediate activations of `Qwen/Qwen3.5-0.8B` for the 8 tokens `[9707, 11, 1879, 0, 358, 1079, 264,
1273]` (`tests/qwen3_5_forward.rs`, ROADMAP row 115): the embedding output, the output of decoder
layers 0, 1, 3 and 23, and the final norm output (float32 `[8, 1024]` each), plus the top-10 token ids
and logit values per position for the whole-sequence forward and for token-at-a-time cached decode
(`[8, 10]`). The full logits (8 x 248320 floats) are not stored.

**Produced by HuggingFace transformers 5.22.1 (Apache-2.0) running the Qwen/Qwen3.5-0.8B checkpoint
(Apache-2.0, Qwen team) in float32 on CPU**, by `scripts/qwen35_reference/forward_ref.py`. The file holds
activations only: no weights, no tokenizer data, no third-party text. The activations are numbers the
reference implementation computed; Fuel does not copy that implementation, it is compared against it.
