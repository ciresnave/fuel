# Qwen3-0.6B decode on the Vulkan backend: where one step goes

ROADMAP item 17 (H12 follow-up), items 20 and 21. Measured 2026-10-21 at `origin/main` b950fa81 plus the instrument in this change, RTX 4070 Laptop GPU (8 GB) via Vulkan (`PreferDiscrete`), release build, through `scripts/gpu-run.ps1`, nothing else running. Tool: `fuel-transformers/examples/qwen3_vulkan_profile.rs`; the per-op table is the new opt-in `fuel_dispatch::compiled::op_profile`.

## Why this was measured

The split-KV attention kernels (0.20.32 to 0.20.38) reach no model: Qwen3 on Vulkan builds attention decomposed and never creates a `FlashAttn` node (`docs/research/split-kv-decode-measurement.md`). Item 20 (capacity-K support in the split kernel plus a Vulkan flash arm in the decode builder) is only worth building if attention is a real share of a decode step, so the step was profiled first.

## Setup (what was and was not run)

- Model: `unsloth/Qwen3-0.6B-GGUF` `Qwen3-0.6B-Q4_K_M.gguf` (the file the M0/M1 yardstick uses), tokenizer `Qwen/Qwen3-0.6B`, greedy argmax, prompt 21 tokens (`PROMPTS[0]` in the chat template), `forward_with_kv_context_persistent` on `fuel_tensor::vulkan_backend::new_device()`, f32 KV cache.
- 1 discarded warm-up run, then **5 runs**; each run = prefill, one first-decode call, then **10 steady decode steps** whose per-op host time is accumulated. Context length during the steady steps: 22 to 32 tokens.
- **What the per-op number is:** host wall time of each kernel launch, summed per `(OpKind, backend)` (`execute_compiled_with_wait_hint`). A kernel that only enqueues work (Vulkan batches into a command buffer) is charged its enqueue cost, **not its device time**; one that blocks (CPU math, a host-to-device or device-to-host copy, a forced flush) is charged all of it. `other` is step wall time minus the sum of launches (planning, waits, flushes, readback). This is not a GPU timeline: no Vulkan timestamp queries exist in the backend, and none were added.
- `MatMul/Cpu` mixes the weight projections with the attention score and value matmuls (same `OpKind`); the table cannot split them. The split below is an estimate, labelled as one.
- Not run: any other model, any context longer than 32 tokens, bf16/f16 weights, the AMD iGPU.

## Result (median over 5 runs; min to max in brackets)

| quantity | value |
|---|---|
| prefill, 21 tokens | 34,187 ms (32,412 to 37,738) |
| first decode call (session build plus 1 token) | 10,471 ms (9,472 to 12,926) |
| **steady decode step** (per-run medians) | **4,271 ms (3,911 to 4,466)**, i.e. about 0.23 tokens/s |

Per-op host time per steady decode step:

| op / backend | ms per step | spread | share of step | launches per step |
|---|---:|---|---:|---:|
| **Copy / Vulkan** | **2,662** | 2,382 to 2,899 | **62.3%** | 397 |
| **MatMul / Cpu** | **1,122** | 1,115 to 1,148 | **26.3%** | 253 |
| other (not inside any launch) | 402 | | 9.4% | |
| Copy / Cpu | 71 | 62 to 83 | 1.7% | 145 |
| WriteSlice / Vulkan | 5.1 | | 0.1% | 56 |
| MulElementwise / Vulkan | 3.7 | | 0.1% | 56 |
| Affine, AddElementwise (Vulkan) | 3.5 | | 0.1% | 56 |
| SoftmaxLastDim, RmsNormLastDim, Rope, Silu, Mul, Add (Cpu) | 2.2 | | 0.05% | 28 to 113 each |
| IndexSelect / Vulkan | 0.1 | | 0.0% | 1 |

Every Vulkan compute op together (WriteSlice, Mul, Affine, Add, IndexSelect) is about 12 ms of host time per step, 0.3%. The step is two things: **397 copies at about 6.7 ms each (62%)** and **253 matmuls running on the CPU (26%)**. Softmax, norms, rope and the activation run on the CPU too, but cost under 3 ms in total at this context length. 253 CPU matmuls per step is 28 layers times 9 (seven weight projections plus the two attention matmuls) plus the logits matmul, so the Q4_K weights are being multiplied on the CPU and the activations bounce between the two devices, which is what the Copy count looks like.

**What the data does not say:** why a copy costs about 6.7 ms (size, a synchronising wait for earlier Vulkan work, a per-copy allocation, or something else: the copy sizes were not recorded), nor which of the 397 copies are weight uploads and which are activations. That is the first thing to find before touching attention.

## Attention's share, and the decision rule for item 20

At this context (22 to 32 tokens) attention is negligible by construction: the softmax is 0.28 ms per step and the 56 attention matmuls are inside the 1,122 ms `MatMul/Cpu` figure, of which they are a tiny part (an estimate from FLOPs, not a measurement: a 1-by-128 by 128-by-32 product per head).

For a longer context the attention cost grows with the cache. A lower bound for the memory traffic alone, **computed, not measured**: 28 layers times 2 (K and V) times 8 KV heads times 128 times 4096 positions times 4 bytes (f32 cache) is about 0.94 GB read per step at 4096 context, about 4 ms at a nominal 250 GB/s for this card. Against the measured 4,271 ms step that is under 0.1%.

**Rule (proposed): item 20 is worth building only when the whole decode step is dominated by attention, i.e. measured attention (scores, softmax and value matmul over all layers, at the target context) is at least about 10% of the step.** The estimate above says that cannot hold until the step itself is below roughly 40 ms at 4096 context, two orders of magnitude under today's. So item 20 is **conditional and, on this evidence, not worth starting now**; the profile should be repeated after the Copy and CPU-matmul costs are gone, at the context lengths the model is meant to serve.

## Copy anatomy (ROADMAP item 21, steps 1 and 2; added 2026-10-21)

Measured at the same ref with two more instruments: `op_profile::set_trace` (every launch of one decode step, in order, with the source and destination storage and the byte count of the first input) and `fuel-vulkan-backend/examples/copy_cost_probe.rs` (bare copy cost by size). Two traced steps (one with a decode-session rebuild before it) gave the same counts and sizes; times below are the range of the two. `Copy/Vulkan` means a `Copy` whose kernel runs on the Vulkan backend, i.e. the source is on the GPU.

### What the 542 copies in a step are

| direction | copies | host time | bytes |
|---|---:|---:|---:|
| Vulkan to host (`Copy/Vulkan`) | 397 | 2,284 to 2,311 ms | 2,390,530,560 |
| host to Vulkan (`Copy/Cpu`) | 144 | 48 to 58 ms | about 0.62 MB |
| host to host | 1 | 0 | 0.6 MB |

By size, Vulkan to host:

| size | copies | total ms | per copy (median / max) | what it is |
|---|---:|---:|---|---|
| 622,329,856 B | 1 | 858 to 890 | | exactly 151,936 x 1,024 x 4 B: the f32 embedding table or its transposed output-head copy (tied embeddings); the consumer was not identified |
| 4 to 12.6 MB | 196 | 1,354 to 1,357 | 7.3 to 7.6 / 12.8 | the f32 weight matrices: 28 layers x 7 (q 8.39 MB, k 4.19, v 4.19, o 8.39, gate 12.58, up 12.58, down 12.58) |
| 64 KB to 1 MB | 56 | 21 to 23 | 0.3 / 1.2 | by count and size (28 x 2; 8 KV heads x 128 x capacity 35 x 4 B = 143 KB) the K and V caches; the consumers were not identified individually |
| at most 64 KB | 144 | 44 to 49 | 0.3 / 1.2 | activations (4,096 B = one 1,024-wide f32 row) |

The weight rows add up: 28 x (8.39 + 4.19 + 4.19 + 8.39 + 12.58 x 3) = 1.76 GB, plus 0.62 GB for the embedding table, is 2.38 GB, against 2.39 GB measured. **The 397 copies are mostly the model's own weights being read back from the GPU to the host on every step**: 197 copies and 2.21 to 2.25 s (about 55% of the step) are the 622 MB matrix and the 196 weight matrices; the K/V caches and activations are 200 copies and about 70 ms together.

The trigger: each weight is Vulkan-resident (the const cache uploads it) while its consumer `MatMul` is stamped to run on the CPU (item 21 step 2 below), so the executor's residency copy brings the weight to the host each step. The launch order in the trace is consistent with that: after four small host-to-device copies (token id, the two rope tables, the mask), the 622 MB matrix is read back first, then per layer the weights in groups with a 4 KB activation copy between them.

### Fixed cost per copy, and bandwidth (`copy_cost_probe`, median of 100, GPU idle)

| bytes | host to device | device to host | device to device + readback |
|---:|---:|---:|---:|
| 4 | 200 us | 193 us | 378 us |
| 4,096 | 199 us | 199 us | 376 us |
| 65,536 | 219 us | 221 us | 414 us |
| 1,048,576 | 450 us | 933 us | 1,352 us |
| 4,194,304 | 1,937 us | 3,312 us | 4,243 us |
| 16,777,216 | 3,640 us | 7,436 us | 10,545 us |
| 67,108,864 | 17,725 us | 42,254 us | 56,020 us |

So the fixed cost of a copy is about 0.2 ms (a submit, a fence wait and staging; both directions), and large device-to-host copies run at about 1.3 to 2.3 GB/s (64 MB in 42 ms), host-to-device about 2.2 to 4.6 GB/s. The decode step is **bandwidth-bound, not fixed-cost-bound**: 397 fixed costs are about 80 ms; the rest is 2.39 GB moved in about 2.3 s, i.e. 1.0 GB/s overall. One probe run per size; the spread of a cell was not measured.

### What the planner believes a copy costs

`SystemTopology::estimate_transfer_ns` (the numbers placement is priced with), against the measured copy times above:

| bytes | planner, device to host | measured, device to host |
|---:|---:|---:|
| 4 | 1.29 ms | 0.19 ms (probe) |
| 12,582,912 | 8.94 ms | 7.3 to 8.5 ms (trace, median 7.3 to 7.6) |
| 622,329,856 | 379 ms | 858 to 890 ms (trace) |

The planner prices a 12.6 MB weight copy about right (and a 4-byte copy about seven times too high, a 622 MB copy 2.3 times too low), so a mis-priced copy does not explain the CPU placement.

### Item 21 step 2: the 253 matmuls

- **How many are Q4_K: none.** All 253 are `OpKind::MatMul` on f32 operands. `lazy_quantized_qwen3.rs` keeps only Q4_0 tensors quantized and dequantizes every other GGML type to f32 at load ("Other GGML dtypes for Linear weights are dequantized to F32"); this Q4_K_M file's Q4_K and Q6_K tensors are all dequantized, which is also why the weights are 8 times their file size and the embedding table is 622 MB. There is no Q4_K matmul path in play on any backend.
- **Why the matmuls run on the CPU: not established.** Ruled out: a missing Vulkan kernel (the Vulkan f32 matmul is registered, and the Vulkan elementwise ops in the same graph run on the GPU); a mis-priced transfer (table above); the planner not seeing the weights' residency at plan time (the plan is built after the const cache upload, and rebuilding the decode session with the same context, `QWEN3_REPLAN=1`, gave the identical 397 / 253 / 145 counts and the same sizes). Not examined: the Vulkan versus CPU matmul cost estimates for these shapes (the cost model's flops and bandwidth terms, and whatever the Judge cache holds on this box). That comparison is the next measurement.

### Where the time could come out (estimates from the numbers above, nothing implemented)

- Stop reading weights back every step (keep a host copy of each immutable weight after its first read, or do not upload a weight whose consumers all run on the host): removes the 197 large copies, **2.21 to 2.25 s of the 4.0 to 4.3 s step (about 55%)**, leaving about 1.8 to 2.0 s.
- Run the matmuls where the weights already are: removes the same copies and the 1.1 s of CPU matmul if the Vulkan kernel is fast for a one-row product; whether it is was not measured.

## What this points at instead (not started)

1. The 397 `Copy/Vulkan` launches per step (62%).
2. The 253 `MatMul/Cpu` launches per step (26%): Q4_K weights multiplied on the CPU. Whether a Vulkan quantized matmul exists for Q4_K, and why the route picker places these on the CPU, was not investigated here.

These are ROADMAP item 21.

## Reproduce

```text
pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release -p fuel-transformers --features dev-tools,vulkan --example qwen3_vulkan_profile -- 5 10
```
Needs the Hugging Face cache (or network) for the GGUF and tokenizer. Takes about 15 minutes (prefill alone is about 34 s per run).
