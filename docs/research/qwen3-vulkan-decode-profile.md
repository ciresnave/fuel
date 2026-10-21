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
- **Why the matmuls run on the CPU: not established.** Ruled out: a missing Vulkan kernel (the Vulkan f32 matmul is registered, and the Vulkan elementwise ops in the same graph run on the GPU); a mis-priced transfer (table above: the estimate varies between processes, 8.9 to 18.6 ms for a 12.6 MB copy, and the stamps did not change); the planner not seeing the weights' residency at plan time (the plan is built after the const cache upload, and rebuilding the decode session with the same context, `QWEN3_REPLAN=1`, gave the identical 397 / 253 / 145 counts and the same sizes). Not examined: the Vulkan versus CPU matmul cost estimates for these shapes (the cost model's flops and bandwidth terms, and whatever the Judge cache holds on this box). That comparison is the next measurement.

### Where the time could come out (estimates from the numbers above, nothing implemented)

- Stop reading weights back every step (keep a host copy of each immutable weight after its first read, or do not upload a weight whose consumers all run on the host): removes the 197 large copies, **2.21 to 2.25 s of the 4.0 to 4.3 s step (about 55%)**, leaving about 1.8 to 2.0 s.
- Run the matmuls where the weights already are: removes the same copies and the 1.1 s of CPU matmul if the Vulkan kernel is fast for a one-row product; whether it is was not measured.

## GPU time per op (Vulkan timestamp queries; ROADMAP items 22 and 23)

`VulkanBackend::set_gpu_timing(true)` makes each batch write one timestamp at its start and one after every dispatch (`BOTTOM_OF_PIPE`), read when the batch is waited on; `gpu_time_snapshot()` returns GPU nanoseconds per op name. The difference of consecutive timestamps is the time between one dispatch retiring and the next retiring, so it is a **serialised view**: dispatches that overlap (no barrier between them) are charged to whichever retires last. The sum over a batch is recorded as `(batch total)`.

**Cross-check against the host clock** (`gpu_time_probe`: n decode-attention dispatches, `h_q = 8`, `Sk = 4096`, in one batch, drained by a readback; median of 5 repeats after 2 warm-ups):

| n | host wall | GPU (batch total) | GPU / wall | GPU per dispatch |
|---:|---:|---:|---:|---:|
| 1 | 1.01 ms | 0.57 ms | 0.56 | 572 us |
| 10 | 1.97 ms | 1.41 ms | 0.71 | 141 us |
| 50 | 6.70 ms | 5.19 ms | 0.78 | 104 us |
| 200 | 24.17 ms | 19.66 ms | 0.81 | 98 us |
| 400 | 47.27 ms | 39.91 ms | 0.84 | 100 us |

The GPU figure never exceeds the wall time and converges towards it as the fixed submit, wait and readback cost amortises; the remaining 16% at n = 400 (about 18 us per dispatch) is the host recording each dispatch (the batch is submitted only after all of it is recorded, so record time and GPU time add rather than overlap). That is agreement between two clocks on the sum, not a check of any single kernel's number.

**Qwen3-0.6B decode step, per-op GPU time** (Q4_K_M GGUF, 21-token prompt, RTX 4070 Laptop, `gpu-run.ps1`, 1 warm-up + 5 runs x 10 timestamp-on steps; median over runs, min to max). Measured with the per-device bit-stable preference of PR #426 applied locally, **not in this PR**: without it these ops run on the CPU and there is almost no GPU work to time (see "Route B" once #426 lands).

| op (kernel name) | ms per step | min to max | share of GPU time | dispatches per step |
|---|---:|---|---:|---:|
| **`matvec`** (the M = 1 matmul) | **82.58** | 79.73 to 94.38 | **95.4%** | 253 |
| `binary_mul_f32` | 1.10 | 1.05 to 1.14 | 1.3% | 141 |
| `rms_norm_last_dim_f32_bytes` | 1.05 | 0.98 to 1.09 | 1.2% | 113 |
| `binary_add_f32` | 0.65 | 0.61 to 0.67 | 0.8% | 84 |
| `write_slice_b4` | 0.47 | 0.43 to 0.52 | 0.5% | 56 |
| `rope_f32_bytes` | 0.29 | 0.28 to 0.33 | 0.3% | 56 |
| `unary_silu_f32` | 0.26 | 0.26 to 0.28 | 0.3% | 28 |
| `softmax_last_dim_f32_bytes` | 0.21 | 0.19 to 0.23 | 0.2% | 28 |
| `affine_f32_bytes` | 0.14 | 0.14 to 0.17 | 0.2% | 28 |
| `index_select_f32_bytes` | 0.005 | | 0.0% | 1 |
| **GPU time, first to last timestamp of each batch** | **86.54** | | | |

The step wall time in the same run was 306 ms (293 to 318, timestamps off) and 334 ms with timestamps on. So:

- **The GPU is busy for about 86 ms of a 306 to 334 ms step, roughly 26 to 28%.** The earlier reading in this note ("the remaining time is the host waiting for the GPU") was wrong: the GPU only accounts for a quarter of it. About 220 ms of each step is host time outside the GPU, and the per-launch host time (about 36 ms in `op_profile`) is only a sixth of that, so roughly 185 ms is executor work that is neither a kernel launch nor GPU execution (graph walk, rebinding, allocation, submission). Where exactly was not measured (ROADMAP item 23).
- **On the GPU, 95% of the time is the M = 1 matmul.** 253 dispatches at 0.33 ms each. The weights it reads are about 2.38 GB per step (1.76 GB of layer weights plus the 0.62 GB vocabulary matrix), so it runs at roughly 29 GB/s, against a nominal 256 GB/s for this card: about 11% of the memory bandwidth, 2.38 GB / 82.6 ms. That figure is computed from the sizes above and the measured time, not read from a counter. Reading 8 times fewer bytes (a native quantized path, the quantized-loader row of PR #426) and a better one-row kernel are the two levers; neither was tried.
- Attention (softmax 0.21 ms, plus the two attention matmuls inside `matvec`) stays negligible at this context length, as item 20 concluded.

## Host time per step: where it goes, and the executor fix (ROADMAP item 23)

After the GPU table the open question was the roughly 185 ms of every step that is neither a kernel launch nor GPU execution. `op_profile::PhaseTimer` (opt-in, same switch as the launch accounting, off by default) names the phases of one decode step: `qwen3_vulkan_profile` prints them. Measured with the per-device bit-stable preference of PR #426 applied locally (not in this PR), Qwen3-0.6B Q4_K_M, 5 runs x 10 steps, median over runs of the per-step average (min to max); phases nest by name, so an outer phase includes the inner ones and rows are not additive.

**Before (executor as on main):** step 343 ms (332 to 354). `realize.loop_total` 370 ms = `execute_work_item` 157 ms (1,959 work items) + **`drain_inflight_vulkan` 155 ms (419 entries per step)** + **`eager_submit_vulkan` 46 ms (401 entries)** + eviction 9 ms + wait for the compiler thread 2 ms. Everything else is small: building the per-token data 10 ms, safety copies 3 ms, planning 0.6 ms, dropping the cache 0.4 ms, the compiler thread 3 ms (concurrent).

**Cause.** About 400 times per step the executor's chunk-boundary logic saw a backend switch that was not one. Work items for `Op::Const` (adopting a weight from the input cache) are always stamped `target_backend = Cpu`, and metadata-only views inherit or default to `Cpu`. The decode graph interleaves them with the Vulkan kernels (each weight `Const` sits just before its matmul), so `current_chunk_backend` flipped Vulkan, Cpu, Vulkan, ... Each "leaving a Vulkan chunk" armed `multi_backend` and submitted the open Vulkan batch (`eager_submit_all_vulkan`); each return to a Vulkan op waited for the in-flight batches (`drain_inflight_vulkan`). The step ran as about 400 tiny submit-and-wait round trips instead of a few batches. (This is also why the per-batch GPU table in the section above summed about 400 batches per step.)

**Change.** `is_backend_neutral(kind)`: `ConstAdopt`, `ViewOf` and `SlotView` items touch no device and no longer take part in the chunk-boundary logic (the `TopologyChanged` check, arming `multi_backend`, the eager submit, and the drain before a Vulkan op). Kernels, copies, moves, writes and `ContiguizeOf` still do. Applied to both executor loops (`realize_inner`, `realize_many_inner`).

**After:** step **251 ms (232 to 295)**, from 343 ms: **27% less, 2.9 to 4.0 tokens/s** (the same build, the same box, back to back). `drain_inflight_vulkan` 79 ms (93 entries per step), `eager_submit_vulkan` 6 ms (63), `execute_work_item` 165 ms (1,959). The 79 ms that remains in the drain is the host waiting for the GPU, which is busy about 86 ms per step (previous section); it is not avoidable by this change because the batch is submitted only after its dispatches are recorded. The 63 remaining eager submits are the 61 evictions of a destroyed input (each in-place KV write).

**Correctness:** `QWEN3_VERIFY=1` against the CPU fixture after the change: max logit difference 9.9e-5 (identical to before), 0 of 64 greedy tokens differ, PASS (one run). This change only moves when the Vulkan batches are submitted and waited. Not tested: graphs that genuinely mix CUDA and Vulkan (the chunk logic exists for them; neutral items never touched a device, so they cannot be what separated two real chunks, but this was not run).

**What remained in the 251 ms** was attributed in the next section: it was not the roughly 1,170 work items that launch nothing (const adoption and views cost about 1 ms of host time in total) but the host round trip of Vulkan auto-contiguize.

## Where the rest of the per-item time went: Vulkan contiguize (ROADMAP item 23, continued)

`execute_work_item` split by work-item kind (`PhaseTimer` names `item.<Kind>` and, inside a kernel item, `kernel.gather_inputs`, `kernel.alloc_output`, `kernel.execute_compiled`), same setup as above (per-device bit-stable preference of #426 applied locally, not in this PR; 5 runs x 10 steps, median over runs of the per-step average). Step 253 ms (234 to 260), before this change:

| phase | ms per step | entries per step |
|---|---:|---:|
| `item.Kernel` | 123 | 805 |
| of which `kernel.gather_inputs` | 86 | 805 |
| of which `kernel.execute_compiled` (the launch) | 25 | 805 |
| of which `kernel.alloc_output` | 8 | 805 |
| `item.ContiguizeOf` | 43 | 156 |
| `item.ViewOf` + `item.ConstAdopt` | 1.0 | 526 + 408 |
| `drain_inflight_vulkan` (host waits on the GPU) | 77 | 93 |

The 1,170 or so items that launch nothing were not the cost (about 1 ms). The cost was `gather_inputs` and `ContiguizeOf`, i.e. `auto_contiguize`: for a Vulkan input whose layout is not contiguous (transposed or permuted views of activations, 123 of them per step) it did **download, CPU `contiguize_cpu`, upload** (the "V.1.B stopgap" in the code), which also forces a flush and a wait on the open batch each time. A separate 3-run measurement with a timer on exactly that path (the box was in a slower stretch, step 298 ms) put it at 152 ms over 123 round trips, about 1.2 ms each, which is where `gather_inputs` plus `ContiguizeOf` went.

**Change.** `vulkan_dispatch::contiguize_vulkan` materializes the layout with the existing `strided_copy_signed_*` kernels (2, 4 and 8-byte elements): no download, no CPU walk, no upload, no forced flush; the dispatch joins the open batch. `auto_contiguize` uses it first and keeps the host round trip as the fallback (other element widths, an empty tensor, an offset past 32 bits, no backend handle).

**Correctness of the kernel path:** `contiguize_vulkan_check` compares it with `contiguize_cpu` byte for byte on 7 layouts (contiguous, transpose, 3-D permute, narrow with a start offset, broadcast, flip = negative stride, flip plus transpose) at each of widths 2, 4 and 8: 21 of 21 identical; widths 1 and 16 correctly decline (23 checks, 0 failures). **Model-level:** `QWEN3_VERIFY=1` against the CPU fixture: max logit difference 9.9e-5 (identical to before this change), 0 of 64 greedy tokens differ, PASS (one run).

**After:** step **122 ms (119 to 129)**, from 253 ms: **2.1x, 4.0 to 8.2 tokens/s**. Prefill 5,994 ms and first decode call 5,545 ms (these vary a lot between runs on this box; no before/after claim). `item.Kernel` 31 ms (`gather_inputs` 4, launch 18, alloc 7), `item.ContiguizeOf` 2.8 ms, `drain_inflight_vulkan` 65 ms (93), evictions 9.5 ms (61), eager submits 4.9 ms (63). With #426 and #428 the decode step has gone from 4,271 ms (0.23 tokens/s) at the start of this investigation to 122 ms (8.2 tokens/s), 35x; #426 is the part not yet merged.

**What is left in the 122 ms:** the host waiting on the GPU (65 ms in the drain, the GPU being busy about 86 ms per step in the earlier table, which was measured before this change), kernel launches 18 ms, per-token data 10 ms, evictions 9.5 ms, output allocation 7 ms. Nothing here has been tried yet: submitting batches earlier so recording overlaps execution, the M = 1 matmul kernel (95% of GPU time), reading fewer bytes (quantized loader).

## The M = 1 matmul kernel (ROADMAP item 24)

By the GPU table the M = 1 `matvec` was 95% of GPU time (82.6 ms of 86.5 ms per step) at about 29 GB/s against a nominal 256 GB/s for this card.

**Cause (read from `matvec.glsl`, then measured):** one workgroup per output column, its 128 threads striding over K and reading `B[k * N + col]`. For the row-major `[K, N]` weights every neighbouring thread reads an address N floats apart: one useful float per cache line.

**Change:** `matvec_cols.glsl`, used by `matmul_f32_bytes` for `m == 1` and `n % 4 == 0`. Threads map to columns: a workgroup owns 32 columns, 8 lanes each load a `vec4` (4 adjacent columns), 32 K-slices walk down K together with four loads in flight per thread, and the slices are summed in a fixed order. A warp now reads consecutive floats of a B row. Other N (the attention shapes, where N is the cache length) keep the original kernel. `set_matvec_cols(false)` selects the old kernel, for the probe only.

**Kernel level** (`matvec_probe`: both kernels against an f64 CPU reference, then GPU timestamps over a batch cycling through distinct weight buffers, so the weights are not in L2; median of 4 timed rounds; a second run after adding a warm-up):

| K | N | role in Qwen3-0.6B | matvec | matvec_cols | speedup | bandwidth old to new |
|---:|---:|---|---:|---:|---:|---|
| 1024 | 1024 | k_proj, v_proj | 43.4 us | 22.5 us | 1.92x | 97 to 186 GB/s |
| 1024 | 2048 | q_proj | 86.9 us | 45.0 us | 1.93x | 96 to 186 GB/s |
| 2048 | 1024 | o_proj | 90.9 us | 45.0 us | 2.02x | 92 to 186 GB/s |
| 1024 | 3072 | gate, up | 168.3 us | 57.7 us | 2.92x | 75 to 218 GB/s |
| 3072 | 1024 | down | 134.2 us | 57.8 us | 2.32x | 94 to 218 GB/s |
| 1024 | 151,936 | lm head | 22.2 ms | 6.1 ms | 3.62x | 28 to 101 GB/s |

An earlier run of the same probe, before the warm-up, gave 6.60x for the LM head (19.0 ms to 2.9 ms, 216 GB/s) and was read slow on the first row for both kernels; the LM-head ratio varies between 3.6x and 6.6x with the GPU's clock mode on this box (ROADMAP item 18). Both kernels were within 1.5e-7 (old) and 2.5e-7 (new) of the f64 reference, relative to the largest output, on every shape. The best new figure, 218 GB/s, is about 85% of nominal.

**Decode step** (the Qwen3 harness on a local build that stacks this change on #429's native contiguize, with the per-device bit-stable preference of #426 applied locally; neither is in this PR; 5 runs x 10 steps, median over runs, min to max):

| | before this kernel | after |
|---|---|---|
| matvec GPU time per step | 82.6 ms, 253 dispatches (`matvec`) | **14.8 ms, 225 dispatches (`matvec_cols`) + 0.4 ms, 28 dispatches (`matvec`)** |
| GPU time per step (first to last timestamp) | 86.5 ms | **20.9 ms** |
| steady decode step | 122 ms (119 to 129) | **91 ms** (82 to 678) |

The slowest of the five per-run medians was 678 ms, most likely a run in the slow GPU mode of ROADMAP item 18 (not separately confirmed; the other per-run medians were not recorded individually), so the median and the minimum are the figures to read; the maximum is real and is not hidden. Against the 86.5 ms baseline the GPU table also gains a `strided_copy_signed` row (1.2 ms, 112 dispatches) that was previously host work (#429). `QWEN3_VERIFY=1` against the CPU fixture: max logit difference **9.2e-5** (9.9e-5 before: the reduction order changed), 0 of 64 greedy tokens differ, PASS (one run).

**Not claimed:** run-to-run bit reproducibility of the new kernel (its K reduction order is fixed by construction, unlike the subgroup-add of the old one, but this was not tested, and the kernel contract still declares the matmul non-bit-stable); half-precision (`matvec_bf16_b`) and quantized weights, which this does not touch; any other GPU. The step is now 91 ms of which the GPU is busy about 21 ms: what is left is host time (recording, launch, waiting) and the batch being submitted only after it is recorded.
## Overlapping recording with GPU execution (ROADMAP item 23, continued)

After the contiguize fix the step was 122 ms and `drain_inflight_vulkan` was 65 ms of it (93 entries per step): the host waiting on the GPU. The executor submits the open Vulkan batch without waiting at each of the 61 evictions (and at chunk boundaries), which should let the GPU run it while the host records the next one. It did not, because of the executor's "Invariant E": before recording any new Vulkan op while batches were in flight, the host waited for all of them. The reason it gave: Vulkan only guarantees that batches BEGIN in submission order, so a later batch could read what an earlier one was still writing. The step therefore ran as record, submit, wait, record, submit, wait.

**Change.** The Vulkan recorder now starts every batch with one compute/transfer to compute/transfer memory barrier (`Recorder::record_inter_batch_barrier`). A pipeline barrier's first synchronization scope includes every command earlier in submission order, across `vkQueueSubmit` calls, so each batch's reads see the earlier batches' writes and its writes wait for their reads. That makes the ordering structural, and the host wait before recording is removed from both executor loops. Waits remain where the host needs a result (D2H reads), before freeing a buffer, and at realize-end.

**Measured** (per-device bit-stable preference of #426 applied locally, not in this PR; Qwen3-0.6B Q4_K_M, 5 runs x 10 steps, median over runs, min to max; same box, baseline is main with #429):

| | before | after |
|---|---|---|
| steady decode step | 122 ms (119 to 129) | **71 ms (70 to 74)**, 1.7x |
| `drain_inflight_vulkan` per step | 65 ms, 93 entries | 22.6 ms, 2 entries |
| GPU time per step | unchanged by this change | 36.7 ms (with the old matvec) |

Stacked with the column-parallel matvec of #430 (a local build before #430 merged): step **62 ms (57 to 64), about 16 tokens/s**, GPU busy 16.4 ms per step, drain 8.5 ms. A confirmation run of this PR's tree after rebasing onto main with #430 merged (with #426's filter still applied locally): step **67 ms (61 to 76)**, GPU 19.1 ms per step, drain 9.2 ms, CPU-fixture check max logit difference 9.2e-5, 0 of 64 greedy tokens differ; the two runs differ by the box's run-to-run noise. With #426, #428, #429 and #430 the step has gone from 4,271 ms to 62 ms, 69x; #426 and #430 were not merged when this was measured.

**Correctness:** `QWEN3_VERIFY=1` against the CPU fixture: max logit difference 9.9e-5 (identical to before this change; 9.2e-5 with #430's kernel), 0 of 64 greedy tokens differ, PASS, in both configurations (one run each). That run crosses about 63 batch boundaries per decode step for 64 steps on each of two prompts.

**Evidence for the ordering.** (1) The rule: a pipeline barrier's first synchronization scope includes the commands earlier in submission order, across `vkQueueSubmit` calls (the Vulkan specification's definition of the scope). (2) The fixture check above: about 63 batch boundaries per decode step, 64 steps on each of two prompts, no divergence. (3) **Khronos synchronization validation** (`VK_LAYER_KHRONOS_validation` from the installed SDK, `validate_sync = true`, core checks left on), with a control that must fail:

| program | build | `SYNC-HAZARD` reports |
|---|---|---|
| `batch_order_probe` (a dependent chain across separate submissions) | with the barrier | **0** |
| `batch_order_probe` | barrier call removed (control) | **10**, read-after-write: "`vkQueueSubmit()`: READ_AFTER_WRITE hazard ... reads VkBuffer ..., previously written by another vkCmdDispatch (from another VkCommandBuffer) ... No sufficient synchronization" |
| the Qwen3 decode (`qwen3_vulkan_profile`, 1 run, 1 step) | with the barrier | **0** |
| the Qwen3 decode | barrier call removed (control) | **20** (10 read-after-write, 10 write-after-read; the layer appears to cap each kind at 10) |

The other 131 validation errors in both decode runs are the same pre-existing pipeline-creation errors in the committed SPIR-V (`VUID-VkShaderModuleCreateInfo-pCode-08740` 50, `VUID-RuntimeSpirv-uniformAndStorageBuffer16BitAccess-06332` 50, `VUID-RuntimeSpirv-None-06278` 20, `VUID-RuntimeSpirv-Workgroup-06530` 6, `...pCode-08737` 5), unrelated to ordering and not touched here; they are a separate finding. (My first attempt ran the layer with core checks disabled; that configuration printed nothing even for a known hazard, which is why the control matters, and is not used.)

(4) On this NVIDIA driver the program also computes the right answer with the barrier removed (`batch_order_probe` passes either way: consecutive submissions did not overlap here), so a wrong number is not what shows the barrier is needed; the validation layer is. Inside ONE batch the GPU does run unsynchronized dispatches concurrently (with the in-batch barriers disabled a dependent chain came out 34x off).

**Not tested:** more than one queue or device (the barrier orders one queue, the recorder uses one); graphs mixing CUDA and Vulkan; a driver that overlaps consecutive submissions (an attempt to run the probe on the AMD iGPU by name gave results identical to the NVIDIA run to three digits, so I do not trust that it ran on a different device and did not use it); a long run for hazards that show only as wrong numbers.

**What is left at 71 ms:** the loop is 65 ms: kernel items 25 ms (launches 15, output allocation 6, input gathering 3), per-token data 9 ms, evictions 6.5 ms, eager submits 3.8 ms, and 22.6 ms waiting at the end for the GPU, which is busy 36.7 ms per step in this configuration (about 16 ms with #430), so the host recording is now the longer pole.

## Allocation churn and the per-token upload (ROADMAP item 23, continued)

With the host no longer waiting on the GPU, two host phases stood out in the profile (per decode step, 805 kernels): `kernel.alloc_output` 7.2 ms and `decode.build_token_data` 10.5 ms. Both traced to the same two causes.

**1. The per-token upload ran a full transient-graph realize per buffer.** `decode.build_token_data` was split into `decode.token_host_compute` (0.02 ms: the RoPE tables and mask are cheap) and `decode.token_upload` (all of the rest). `upload_host_buffer_to_device` built a transient `Op::Const -> Op::Copy` graph and ran `PipelinedExecutor::realize_many` on it for each of the four per-token buffers. On Vulkan it now calls `upload_bytes_handle` directly (the same staging copy the realized `Op::Copy` performs).

**2. The buffer recycler destroyed and recreated the step's temporaries every step.** Instrumenting `take_or_alloc_device_buffer` showed the 3 ms was not allocation: the first allocation of a step, holding the pool lock, trimmed the 2,880-byte bucket from hundreds of buffers to the per-bucket depth of 4, and every `vkDestroyBuffer` is about a millisecond. The 2,880-byte bucket then missed 7,140 times against 2,100 hits in 7 steps. The depth is now `recycler_bucket_depth(size)`: as many buffers as fit in 16 MiB, between 4 and 1,024 (large buffers keep 4), and trimmed buffers are destroyed after the lock is released. The pool's total-bytes cap (512 MB) and bucket-count cap are unchanged.

**Measured** (Qwen3-0.6B Q4_K_M decode on Vulkan, RTX 4070, #426's per-device preference applied locally and not in this PR, 3 runs x 10 steps, median step, baseline and new binaries run alternately three times):

| | baseline (main) | this change |
|---|---|---|
| step, 3 alternating pairs | 62, 59, 59 ms | **43, 44, 45 ms** |
| `kernel.alloc_output` per step | 7.2 ms | 1.3 ms |
| `decode.token_upload` per step | about 10.5 ms | 1.7 ms |

About 1.35x on the step, about 22 tokens/s. A first, non-alternating comparison was unusable (the box was loaded by other work: the GPU batch time moved from 14 to 19 ms between runs), which is why the table is from alternating runs.

**Correctness:** `QWEN3_VERIFY=1` against the CPU fixture: max logit difference 9.2e-5, 0 of 64 greedy tokens differ, PASS (one run, on the tree with both changes). **Cost:** the recycler can now hold up to its existing 512 MB cap in more, smaller buffers; worst case, measured with `recycler_pool_probe` (RTX 4070, 64 distinct sizes from 2 KiB to 24 MiB, as many buffers per size as the depth allows, 21,620 buffers, 1,260 MiB, all freed): 1,260 MiB parked until the next allocation, 511 MiB after it (the 512 MB cap is applied at allocation time, so it can be exceeded between a mass free and the next allocation). The pre-change depth was not run on the same probe.

**Not tried:** batching the four uploads into one submission. Measured inside `upload_bytes`: the copy submission is 0.3 to 0.5 ms of the upload, the rest was the pool trim, so batching would save about 1 ms and was not kept.

## False backend switches, deferred eviction and epoch-gated buffer reuse (ROADMAP item 23, continued)

After the allocation fixes the step was 43-52 ms and `realize.evict_destroyed_input` was 5.8 ms (61 per step). Timing its parts showed the cost was not the eviction itself: `eager_submit_all_vulkan` scans the whole storage cache for Vulkan backends (2.2 ms per step over 123 calls) and submits (about 90 us per real `vkQueueSubmit`, which also creates a fence and a command pool). A temporary log of every chunk switch showed where the submits came from: **910 `Vulkan -> Cpu` switches in two steps, every one at an `item.ContiguizeOf`.** `ContiguizeOf` is stamped with a `Cpu` placeholder `target_backend` (it runs on its input's backend), so each of its 156 appearances per step looked like a backend switch, armed `multi_backend`, and submitted the open batch. ROADMAP item 23's earlier neutral-item fix (#428) covered `ConstAdopt`/`ViewOf`/`SlotView` only.

**Changes.** (1) `ContiguizeOf` is backend-neutral for chunking. (2) A Vulkan kernel item arms the async path (`multi_backend`) itself, so a pure-Vulkan realize keeps deferred eviction instead of the blocking per-eviction flush. (3) A deferred eviction retains the evicted buffer on the OPEN batch (`VulkanBackend::retain_in_open_batch`, freed post-fence once the batch is submitted) and on every in-flight batch, and no longer submits; the open batch is submitted once it holds 48 dispatches (`EVICT_SUBMIT_MIN_DISPATCHES`). (4) **Buffer reuse is epoch-gated.** The recycler stamps each parked buffer with the batch epoch it was freed in (bumped at every submission); a buffer freed in the open batch is not handed out again until a batch boundary. Larger batches exposed this: with (1)-(3) alone the decode was wrong (max logit difference 30.08, 18 of 64 tokens differ) and the Khronos validation layer reported `SYNC-HAZARD-WRITE-AFTER-READ` on output buffers (binding 2) previously read by another dispatch in the same batch. The recorder tracks write-then-read inside a batch but not write-after-read, and a freed-then-reallocated buffer is exactly that. Before this change a batch ended at every eviction, and the next batch's opening barrier (item 23's inter-batch barrier) hid it; the hazard was latent for any realize whose frees and allocations land in one batch.

**Evidence the hazard fix is the cause.** Same binary, submit threshold swept: 1, 4 correct; 12, 24, 48 wrong (identical wrong logits, deterministic) before the epoch gate; all thresholds correct after it, and validation reports 0 `SYNC-HAZARD` (the 11 known SPIR-V errors remain). A sabotage test (`a_buffer_freed_in_the_open_batch_is_not_reusable_until_a_batch_boundary`, `<` changed to `<=`) reddens.

**Measured** (Qwen3-0.6B Q4_K_M, RTX 4070, #426's per-device preference applied locally and not in this PR; step median per binary run of 3x10 steps; the box was shared and loaded, 60-66% CPU from other lanes, so only runs inside a quiet window are quoted):

| submit threshold (dispatches) | step |
|---|---|
| 16 / 32 / 48 / 200 / never | 40 / 38 / **35** / 40 (min 35, one slow outlier) / 46 ms |

Alternating base (main) and new binaries, three pairs, quiet-window medians: **base 54, 52, 43-52 ms; new 36, 32-35, 30-31 ms**, about 1.5x, about 30 tokens/s. Other runs in the same sessions were 3-5x slower for both binaries (every host phase scaled together, so host CPU contention, not this change). `QWEN3_VERIFY=1`: max logit difference 9.2e-5, 0 of 64 tokens differ, PASS, at every threshold after the epoch gate (one run each).

**Cost / not tested:** buffer reuse now waits for the next batch boundary, so the pool holds more buffers at once (pool VRAM not re-measured on a decode; `recycler_pool_probe` covers the worst case of the cap). `cargo test -p fuel-dispatch -p fuel-vulkan-backend --features fuel-dispatch/vulkan` passes (816 lib tests, 0 failed). Not tested: CUDA+Vulkan mixed graphs (the multi-backend deferred path now also runs for them, with the open-batch retain), more than one Vulkan device, models other than Qwen3-0.6B. The pure-Vulkan realize used to take the blocking single-device eviction path; it now takes the deferred one, so any other single-device Vulkan graph changes path.

## What this points at instead (not started)

1. The 397 `Copy/Vulkan` launches per step (62%).
2. The 253 `MatMul/Cpu` launches per step (26%): Q4_K weights multiplied on the CPU. Whether a Vulkan quantized matmul exists for Q4_K, and why the route picker places these on the CPU, was not investigated here.

These are ROADMAP item 21.

## Reproduce

```text
pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release -p fuel-transformers --features dev-tools,vulkan --example qwen3_vulkan_profile -- 5 10
```
Needs the Hugging Face cache (or network) for the GGUF and tokenizer. Takes about 15 minutes (prefill alone is about 34 s per run).
