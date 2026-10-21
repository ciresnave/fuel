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

`SystemTopology::estimate_transfer_ns` (the numbers placement is priced with) comes from a transfer probe the topology runs lazily at startup, so it differs between processes. Two processes on the same box, same day, against the measured copy times above:

| bytes | planner, device to host (process A / process B) | measured, device to host |
|---:|---:|---:|
| 4 | 1.29 / 1.60 ms | 0.19 ms (probe) |
| 12,582,912 | 8.94 / 18.57 ms | 7.3 to 8.5 ms (trace, median 7.3 to 7.6) |
| 622,329,856 | 379 / 841 ms | 858 to 890 ms (trace) |

So the planner's price of a 12.6 MB weight copy is between about right and 2.4 times too high depending on the process, and a 4-byte copy is 7 to 8 times too high; it is never too LOW for a weight copy except the 622 MB one in process A (2.3 times low). In neither process does an under-priced weight copy explain the CPU placement: in process B the planner believed each weight copy cost 18.6 ms and still stamped the matmuls to the CPU. (Process B is the run that produced the 3,707 ms step and the 2,353 ms of Vulkan-to-host copy time in the repeat check of the refactored example; the counts and sizes were again 542 / 397 / 2,390,761,728 B.)

### Item 21 step 2: the 253 matmuls

- **How many are Q4_K: none.** All 253 are `OpKind::MatMul` on f32 operands. `lazy_quantized_qwen3.rs` keeps only Q4_0 tensors quantized and dequantizes every other GGML type to f32 at load ("Other GGML dtypes for Linear weights are dequantized to F32"); this Q4_K_M file's Q4_K and Q6_K tensors are all dequantized, which is also why the weights are 8 times their file size and the embedding table is 622 MB. There is no Q4_K matmul path in play on any backend.
- **Why the matmuls run on the CPU: established (route B, below).** The planner never compares their costs at all: a filter removes the Vulkan candidate first.

## Route B: why the planner picks the CPU, the fix, and before and after (ROADMAP item 21 step 3)

### Root cause (found with a temporary candidate-set print in `compile_plan`, not committed)

For every one of these `MatMul` nodes the candidate set the planner enumerates, before any filtering, is two entries:

| candidate | kernel source | `bit_stable_on_same_hardware` |
|---|---|---|
| `Vulkan@Vulkan { gpu_id: 1 }` | `vulkan-slang` | **false** |
| `Cpu@Cpu` | `portable-cpu` | true |

After `default_chain` the set is `{ Cpu }`. The step that removes the Vulkan candidate is `BitStablePreferenceFilter` (soft, `min_remaining = 1`): when any bit-stable candidate exists it drops every non-bit-stable one, **across devices, before cost ranking**. The Vulkan f32 matmul is honestly non-bit-stable (`docs/kernel-contracts/vulkan/matmul.fkc.md`: "scheduler-dependent FADD/subgroup order, non-associative f32", audited `none(reason)`), so it can never be compared with the CPU kernel on cost, transfer or throughput. The same filter explains `RmsNormLastDim`, `Rope`, `SoftmaxLastDim` and `SiluElementwise` on the CPU in the first profile. It is not a cost-model, transfer-pricing, residency or missing-kernel problem (all four were ruled out above), and the verification ledger cannot fix it for matmul because the claim is false, not unaudited. (`seed_cuda_ledger.rs` documents the same mechanism for CUDA kernels, where the claim is true and the ledger earns it.)

### The change

`BitStablePreferenceFilter` now drops a non-bit-stable candidate only when a bit-stable candidate exists **on the same device**. The preference chooses a kernel on a device; it no longer chooses the device. If nothing is bit-stable anywhere the filter still returns nothing (the chain soft-skips it, as before). A hard determinism requirement is untouched: `PrecisionRequirement::require_bit_stable` (`PrecisionFloorFilter`, strict reproducibility mode) still removes non-bit-stable candidates everywhere.

**This is a default-behaviour change and a policy call.** With it, a GPU kernel declared non-bit-stable can win a placement on cost, so such a graph is no longer bit-identical from run to run in the last bits (for matmul: summation order across subgroups). Callers who need bit-reproducibility already have the strict mode; whether the default should stay "deterministic CPU unless the GPU kernel is bit-stable" is for the project owner to rule on.

### Before and after (same harness, same box, `gpu-run.ps1`, Qwen3-0.6B Q4_K_M, 21-token prompt, 5 runs each, median with min to max)

| | before | after |
|---|---|---|
| steady decode step | 4,271 ms (3,911 to 4,466) | **329 ms (313 to 337)** |
| decode speed | 0.23 tokens/s | **3.04 tokens/s (13x)** |
| prefill, 21 tokens | 34,187 ms (32,412 to 37,738) | 6,764 ms (6,253 to 7,251) |
| first decode call (session build) | 10,471 ms (9,472 to 12,926) | 6,096 ms (5,928 to 6,527) |
| copies per decode step | 542 (397 Vulkan to host, 2.39 GB) | **5** (4 host to Vulkan, 1,168 B; 1 Vulkan to host, the 607,744 B logits row) |
| launches per decode step | 1,330 | 793 |
| `MatMul` | 253 on the CPU | 253 on Vulkan |

Every op that ran on the CPU now runs on Vulkan (the table of the traced step has no `/Cpu` compute op left). Host launch time of all launches is about 36 ms of the 329 ms step; the remaining 293 ms is time the host spends waiting (GPU execution plus the logits readback), which this instrument cannot split.

### Correctness

`QWEN3_VERIFY=1` runs the M0/M1 yardstick comparison on the Vulkan device: both fixture prompts, 32 greedy decode steps each, every step through `compare_step` at the f16-or-quantized tier (relative bound 2e-2) against the checked-in CPU reference. Result: **max absolute logit difference over the reference top-16 of 9.9e-5, 0 of 64 greedy tokens differ, no divergent step: PASS** (one run). That the GPU path is within tolerance of the CPU reference is shown; that it is bit-reproducible run to run is not claimed (and is not expected for the matmul).

### What is still slow, and what is not claimed

3 tokens/s is far below what a 0.6B model should reach on this card. A rough floor from bandwidth alone: the f32 weights are about 2.4 GB, so one step reads at least that, about 10 ms at a nominal 250 GB/s, against 293 ms of waiting. The remaining gap is unmeasured: the Vulkan backend has no timestamp queries, so per-kernel device time is unknown, and the one-row (M=1) matmul path was not examined. Also unchanged: the loader dequantizes every GGML type except Q4_0 to f32 (the 622 MB matrix and 8x weight size), and the item 20 rule (attention at least about 10% of the step) is not yet met at 4096 context by the bound in the section above (about 4 ms against 329 ms is 1.2%).

## Reproduce

```text
pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release -p fuel-transformers --features dev-tools,vulkan --example qwen3_vulkan_profile -- 5 10
```
Needs the Hugging Face cache (or network) for the GGUF and tokenizer. Takes about 15 minutes before the fix (prefill alone was about 34 s per run), about 5 minutes after. `QWEN3_VERIFY=1` first runs the CPU-fixture comparison.
