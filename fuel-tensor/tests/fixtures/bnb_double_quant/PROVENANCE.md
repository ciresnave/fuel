# bnb double-quant NF4 fixture

Real bitsandbytes output, cut down to 16 rows so the repository stays small.

| File | What it is | Source |
|---|---|---|
| `bnb_double_quant_k_proj_16rows.safetensors` | The first 16 rows (of 1024) of `model.layers.1.self_attn.k_proj` from an NF4, double-quantized checkpoint: `.weight` (8,192 packed bytes), `.weight.absmax` (256 U8 codes), `.weight.nested_absmax` (the first of 64 F32 scales), `.weight.nested_quant_map`, `.weight.quant_map`, `.weight.quant_state.bitsandbytes__nf4` | `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`, `model.safetensors` (576,210,569 bytes), read on 2026-10-19 by HTTP range requests; config says `bnb_4bit_quant_type: nf4`, `bnb_4bit_use_double_quant: true`; repo licence apache-2.0 |
| `orig_k_proj_16rows_bf16.bin` | The same 16 rows of the unquantized weight, bf16 little-endian, 16 x 1024 | `Qwen/Qwen3-0.6B`, `model.safetensors`, tensor `model.layers.1.self_attn.k_proj.weight`, read the same day by range request; licence apache-2.0 (`LICENSE-Qwen3-0.6B-Apache-2.0.txt`, fetched from the model repository) |

## What was changed from the originals

- Only the leading 16 rows were kept. A 256-code nested block covers 256 first-level absmax values = 4 rows of 1024 (64 per row), so 16 rows span exactly 1 nested block and `nested_absmax` is the first F32 of the original 64; `nested_offset` (0.07291832566261292, the mean over the whole layer) is unchanged.
- The JSON `shape` in the `quant_state` tensor was patched from `[1024, 1024]` to `[16, 1024]` to match the slice. Every other JSON field is the checkpoint's (`quant_type: nf4`, `blocksize: 64`, `dtype: bfloat16`, `nested_blocksize: 256`, `nested_dtype: float32`, `nested_offset`).
- The safetensors container was rewritten by hand around those tensors (tensor bytes are verbatim).

## Why this fixture exists

It is the oracle the double-quant loader is checked against: dequantizing the slice through Fuel's NF4 op must reproduce the original bf16 weights to NF4's ordinary 4-bit error (relative RMS about 0.09 on this slice), which a wrong absmax decode or a wrong nibble order cannot. Measured while building the fixture: decoding with the bitsandbytes order (even-k code in the UPPER nibble) gives 0.091; Fuel's kernel order (LOWER nibble) gives 1.42. bitsandbytes' own reference packs `(q[::2] << 4) | q[1::2]` (`bitsandbytes/backends/default/ops.py` at `833649043474794b8fe7a4136e0c40faf077b2e0`).
