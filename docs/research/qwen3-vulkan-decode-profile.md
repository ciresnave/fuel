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

## What this points at instead (not started)

1. The 397 `Copy/Vulkan` launches per step (62%).
2. The 253 `MatMul/Cpu` launches per step (26%): Q4_K weights multiplied on the CPU. Whether a Vulkan quantized matmul exists for Q4_K, and why the route picker places these on the CPU, was not investigated here.

These are ROADMAP item 21.

## Reproduce

```text
pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release -p fuel-transformers --features dev-tools,vulkan --example qwen3_vulkan_profile -- 5 10
```
Needs the Hugging Face cache (or network) for the GGUF and tokenizer. Takes about 15 minutes (prefill alone is about 34 s per run).
