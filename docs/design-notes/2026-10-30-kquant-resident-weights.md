# K-quant weights stay quantized from GGUF to `QMatMul` (ROADMAP row 124)

Date: 2026-10-30. Status: design, then code in the same PR. Ordered first by the PM
(2026-10-30) because the Qwen3.8-27B route (rows 117 to 122) cannot fit any rig while a
Q4_K_M file is expanded to f32 at load. Evidence tags: READ = read at the stated ref,
ARITHMETIC = computed from stated inputs, NOT MEASURED = no run.

## The defect (READ, `origin/main@e3ff3427`)

- `WeightStorage` (`fuel-tensor/src/lazy.rs`) has `F32`, `BF16`, `Q4_0`, `Awq`, `WithLoRA`.
- `QuantizedLlama3Model::from_gguf`'s `load_weight` keeps a tensor quantized only for
  `GgmlDType::Q4_0`; every other dtype goes through `dequant_bytes_to_f32` and is transposed
  into an `F32` weight. The other `lazy_quantized_*` loaders follow the same policy.
- ARITHMETIC: a 27B-parameter Q4_K_M file (about 16 GB) becomes about 108 GB of f32.

## The change

1. **One new variant, not three.** `WeightStorage::Ggml { qtype: QuantType, words: Arc<[u32]>,
   bytes_len, in_features, out_features }`: the same shape as `Q4_0`, plus the block format.
   `Q4_0` stays as it is (158 constructing or matching sites; folding it in is a separate,
   mechanical change with no benefit for this row). Graph side: the same 1-D `U32` const and
   `Tensor::qmatmul(&w, qtype, k, n)` that `Q4_0` uses; nothing new in the IR.
2. **Loader policy** (`from_gguf`, shared helper so the later `qwen35` loader of row 118
   reuses it): `Q8_0`, `Q4_K`, `Q6_K` stay resident as `Ggml`. Admissibility is
   `in_features % elements_per_block == 0` (32 for `Q8_0`, 256 for the K-quants); a tensor
   that fails it keeps the old f32 path and the loader records that, because silently taking
   the 4x path is the defect being removed. Other dtypes (f16, bf16, f32, and block formats
   with no CPU `QMatMul` kernel) are unchanged.
3. **`WeightStorage::resident_bytes()`** reports what the weight holds in memory. It is the
   instrument the red test asserts on, and a later gate can sum it per model.
4. **Reach.** The CPU dispatch has `qmatmul_q8_0_f32`, `qmatmul_q4_k_m_f32` and
   `qmatmul_q6k_f32` (READ, `fuel-dispatch/src/dispatch.rs`). CUDA has the MMVQ binding of
   row 105 (#527, unmerged) for Q4_0, Q8_0, Q4_K; Vulkan fuses Q4_0 only (row 119). This row
   does not change those; it makes the CPU and CUDA kernels reachable from a loaded model.

## Red first

`fuel-transformers/tests/gguf_resident_quantized.rs`: a hand-built GGUF (tensor infos only;
`from_gguf` takes its config from the caller) of a one-layer llama with `hidden = 256`,
`intermediate = 256`, projection weights quantized by `fuel_quantized`. For each of Q4_K,
Q6_K and Q8_0 it asserts (a) every projection is `WeightStorage::Ggml` with that `qtype`,
(b) the sum of `resident_bytes()` over projections equals the bytes in the file for those
tensors, and is under a third of their f32 size, (c) logits equal the logits of the same file
loaded with the old dequantize-to-f32 policy to a calibrated tolerance (so the quantized
path computes the same function; the tolerance comes from sabotage, not inheritance).
The red state is the unmodified loader: (a) and (b) fail because the weights are `F32`.

## Not covered

- Embeddings and a tied head stay f32 (row 122: a 248320 x 5120 table is 5.1 GB of f32).
- Vulkan reach (row 119), the `qwen35` tensor layout (row 118), MoE expert tensors.
- Models other than `QuantizedLlama3Model` keep their own loaders until moved onto the shared
  helper; this row converts the helper and the llama loader and lists the others.
- No timing or memory measurement of a real GGUF; resident bytes are asserted on a synthetic
  file.
