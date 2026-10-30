# qwen3_5_0_8b_gdn_mixer_reference.safetensors

The input and output of the Gated DeltaNet token mixer (`linear_attn`) of layers 0 and 1 of
`Qwen/Qwen3.5-0.8B`, float32 `[8, 1024]` each, for the 8 tokens `[9707, 11, 1879, 0, 358, 1079, 264,
1273]` (`tests/qwen3_5_gdn_mixer.rs`, ROADMAP row 115).

**Produced by HuggingFace transformers 5.22.1 (Apache-2.0) running the Qwen/Qwen3.5-0.8B checkpoint
(Apache-2.0, Qwen team) in float32 on CPU**, by `scripts/qwen35_reference/mixer_ref.py`. The file holds
activations only: no weights, no tokenizer data, no third-party text. The activations are numbers the
reference implementation computed; Fuel does not copy that implementation, it is compared against it.
