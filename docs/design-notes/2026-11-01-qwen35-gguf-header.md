# Qwen3.5 GGUF (`qwen35`): what a real file contains

ROADMAP row 118, step one: read the header of an installed `qwen35` file and write down what is
there, assuming nothing. Nothing in this note is a mapping decision; that is step two.

**File.** `unsloth/Qwen3.5-4B-GGUF`, `Q6_K`, Ollama blob `sha256-fdedd781...` (3,525,956,768 bytes,
starts with `GGUF`). Opened read-only; the header was parsed with a 60-line Python reader (no model
was loaded and no GPU used). GGUF v3, 426 tensors, 46 metadata keys. Other installed `qwen35` files
(unsloth Qwen3.5-9B, three 4B fine-tunes) were not opened; each may differ.

## Metadata keys (`general.architecture = qwen35`)

| key | value | note |
| --- | --- | --- |
| `qwen35.block_count` | 32 | |
| `qwen35.embedding_length` | 2560 | |
| `qwen35.feed_forward_length` | 9216 | |
| `qwen35.context_length` | 262144 | |
| `qwen35.attention.head_count` / `head_count_kv` | 16 / 4 | |
| `qwen35.attention.key_length` / `value_length` | 256 / 256 | head dim 256, not `embedding / heads` (160) |
| `qwen35.attention.layer_norm_rms_epsilon` | 9.99999997e-7 | |
| `qwen35.rope.freq_base` | 10,000,000 | |
| `qwen35.rope.dimension_count` | 64 | partial rotary: 64 of 256 dims |
| `qwen35.rope.dimension_sections` | `[11, 11, 10, 0]` | multi-section rope, sums to 32 = 64 / 2 |
| `qwen35.full_attention_interval` | 4 | |
| `qwen35.ssm.conv_kernel` | 4 | depthwise conv width |
| `qwen35.ssm.state_size` | 128 | per-head state width |
| `qwen35.ssm.group_count` | 16 | q/k heads of the DeltaNet |
| `qwen35.ssm.time_step_rank` | 32 | value heads |
| `qwen35.ssm.inner_size` | 4096 | 32 value heads x 128 |
| `tokenizer.ggml.model` / `pre` | `gpt2` / `qwen35` | EOS 248046, padding 248055 |
| `general.file_type` | 18 | Q6_K |

No `output.weight` tensor exists: the output head is tied to `token_embd.weight`.

## Tensors

Types in this file: F32 x177, Q6_K x177, Q8_0 x72. **There is no Q4_K tensor** (this is a Q6_K file;
a Q4_K_M file is a different artifact and still needs reading).

Global: `token_embd.weight [2560, 248320] Q6_K`, `output_norm.weight [2560] F32`.

Every block `N` (32): `attn_norm.weight`, `post_attention_norm.weight` (F32 `[2560]`),
`ffn_gate.weight [2560, 9216]`, `ffn_up.weight [2560, 9216]`, `ffn_down.weight [9216, 2560]` (Q6_K).

Layers 3, 7, 11, 15, 19, 23, 27, 31 (8, one in four: full attention) also carry `attn_q.weight
[2560, 8192]`, `attn_k.weight [2560, 1024]`, `attn_v.weight [2560, 1024]`, `attn_output.weight
[4096, 2560]` (Q6_K) and `attn_q_norm.weight [256]`, `attn_k_norm.weight [256]` (F32). The
`attn_q` output (8192) is twice 16 heads x 256 = 4096, which is consistent with a gated query (not verified here).

The other 24 layers (Gated DeltaNet) carry `attn_qkv.weight [2560, 8192]`, `attn_gate.weight
[2560, 4096]` (Q6_K), `ssm_alpha.weight [2560, 32]`, `ssm_beta.weight [2560, 32]`,
`ssm_out.weight [4096, 2560]` (Q8_0), `ssm_conv1d.weight [4, 8192]`, `ssm_a [32]`, `ssm_dt.bias
[32]`, `ssm_norm.weight [128]` (F32). The conv channel count 8192 = 2 x (16 x 128) + 32 x 128
(q, k, v concatenated), which matches `attn_qkv`'s 8192.

GGUF stores each matrix as `[in, out]`, the reverse of the HuggingFace `[out, in]`.

## What this means for the other rows

- **Row 119 (Vulkan fused quantized matmul):** the weights that dominate this model are **Q6_K**
  (177 of the 249 quantized tensors, including the 248,320 x 2,560 tied embedding/output head) with Q8_0 for the
  small DeltaNet projections. A Q4_K_M-only fused kernel would not touch it; Q6_K matters first for
  this file, Q8_0 second.
- **Row 124 step 3 / the loader:** the loader must handle per-tensor mixed types in one file
  (Q6_K + Q8_0 + F32), tied embeddings, and 8192-wide gated q.
- **Not established here:** the tensor-name mapping to Fuel's `Qwen3_5Model` weights, the exact
  q/k/v split inside `attn_qkv`, the order of the gate halves in `attn_q`, and whether an `mmproj`
  (vision) file is needed (this file has no vision tensors).
