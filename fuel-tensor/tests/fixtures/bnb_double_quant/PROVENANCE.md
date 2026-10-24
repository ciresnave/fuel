# bnb double-quant NF4 fixture

Real bitsandbytes output, cut down to 16 rows so the repository stays small.

| File | What it is | Source |
|---|---|---|
| `bnb_double_quant_k_proj_16rows.safetensors` | The first 16 rows (of 1024) of `model.layers.1.self_attn.k_proj` from an NF4, double-quantized checkpoint: `.weight` (8,192 packed bytes), `.weight.absmax` (256 U8 codes), `.weight.nested_absmax` (the first of 64 F32 scales), `.weight.nested_quant_map`, `.weight.quant_map`, `.weight.quant_state.bitsandbytes__nf4` | `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`, `model.safetensors` (576,210,569 bytes), read on 2026-10-19 by HTTP range requests; config says `bnb_4bit_quant_type: nf4`, `bnb_4bit_use_double_quant: true`; repo licence apache-2.0 |
| `orig_k_proj_16rows_bf16.bin` | The same 16 rows of the unquantized weight, bf16 little-endian, 16 x 1024 | `Qwen/Qwen3-0.6B`, `model.safetensors`, tensor `model.layers.1.self_attn.k_proj.weight`, read the same day by range request; licence apache-2.0 (`LICENSE-Qwen3-0.6B-Apache-2.0.txt`, fetched from the model repository) |

## Digests of the committed bytes (added 2026-10-24)

SHA-256 of the blobs as committed (`git show <ref>:<path> | sha256sum`, so line-ending conversion cannot change them):

| File | Bytes | SHA-256 |
|---|---|---|
| `bnb_double_quant_k_proj_16rows.safetensors` | 10,382 | `1d6b7975dada0c24563ad630e944d7677714d40074c962ff462e31042026e971` |
| `orig_k_proj_16rows_bf16.bin` | 32,768 | `1d91c8f8637e36ff5e0f439b459f228e55339150663302f2f58fcce547937460` |
| `LICENSE-Qwen3-0.6B-Apache-2.0.txt` | | `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` (equal to the `LICENSE` file served by `Qwen/Qwen3-0.6B` on 2026-10-24) |

## Verified against the hub on 2026-10-24

Re-read by HTTP range request, not from the earlier session's memory:

- `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`: tensor `model.layers.1.self_attn.k_proj.weight` is BF16 `[1024, 1024]`; its first 32,768 bytes are **byte-identical** to `orig_k_proj_16rows_bf16.bin`.
- `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`, revision `e86f5289a2ca40ab7114071959dc2079cf97cb3c`, `model.safetensors`: for `model.layers.1.self_attn.k_proj.weight` the fixture's `.weight` (first 8,192 of 524,288 bytes), `.absmax` (first 256 of 16,384), `.nested_absmax` (first 4 of 256), `.nested_quant_map` (1,024) and `.quant_map` (64) are **byte-identical** to the hub's. `.quant_state.bitsandbytes__nf4` differs, as documented below: the fixture's JSON has `shape` patched to `[16, 1024]` (170 bytes against the hub's 172).
- Licences as the hub publishes them today (model-card `license` field and tag): `Qwen/Qwen3-0.6B` **apache-2.0**; `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` **apache-2.0** (base model `Qwen/Qwen3-0.6B`). Both claims in the table above hold.

## What was changed from the originals

- Only the leading 16 rows were kept. A 256-code nested block covers 256 first-level absmax values = 4 rows of 1024 (64 per row), so 16 rows span exactly 1 nested block and `nested_absmax` is the first F32 of the original 64; `nested_offset` (0.07291832566261292, the mean over the whole layer) is unchanged.
- The JSON `shape` in the `quant_state` tensor was patched from `[1024, 1024]` to `[16, 1024]` to match the slice. Every other JSON field is the checkpoint's (`quant_type: nf4`, `blocksize: 64`, `dtype: bfloat16`, `nested_blocksize: 256`, `nested_dtype: float32`, `nested_offset`).
- The safetensors container was rewritten by hand around those tensors (tensor bytes are verbatim).

## Why this fixture exists

It is the oracle the double-quant loader is checked against: dequantizing the slice through Fuel's NF4 op must reproduce the original bf16 weights to NF4's ordinary 4-bit error (relative RMS about 0.09 on this slice), which a wrong absmax decode or a wrong nibble order cannot. Measured while building the fixture: decoding with the bitsandbytes order (even-k code in the UPPER nibble) gives 0.091; Fuel's kernel order (LOWER nibble) gives 1.42. bitsandbytes' own reference packs `(q[::2] << 4) | q[1::2]` (`bitsandbytes/backends/default/ops.py` at `833649043474794b8fe7a4136e0c40faf077b2e0`).
