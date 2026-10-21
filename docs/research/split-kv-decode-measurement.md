# Split-KV decode attention on Vulkan: what was measured

ROADMAP item 17, row **H12** (FlashAttention `num_splits_heuristic`). Measured 2026-10-20, RTX 4070 Laptop GPU via Vulkan (`PreferDiscrete`), release build, `scripts/gpu-run.ps1`, nothing else running. Probe: `fuel-vulkan-backend/examples/flash_attn_split_probe.rs` (an example, not an `#[ignore]` test, to keep the ignore ceiling).

## What the production kernel does

The Vulkan attention forward that runs today is `fuel-kernels-source/kernels/flash_attn_f32.glsl` (not the Slang file the harvest row cited): one workgroup per `(b, h_q, q_i)`, scores for the whole key range in shared memory (`Sk <= 4096`), and a V-accumulate loop that walks every key serially per output element. A decode step (`sq = 1`) therefore launches `b * h_q` workgroups however long the cache is.

## What was added

`flash_attn_split_f32.glsl` + `flash_attn_combine_f32.glsl`, driven by `VulkanBackend::flash_attn_f32_split_bytes(.., splits)`:
- pass 1 runs `(splits, 1, b*h_q*sq)` workgroups, each over a chunk of `ceil(Sk / splits)` keys (at most 1024), writing an unnormalised partial `o[d]`, the chunk max `m` and sum `l`;
- pass 2 merges the partials of a row in fixed split order: `M = max m`, `L = sum l*exp(m-M)`, `out = sum o*exp(m-M) / L`. Deterministic for a given `splits`; it reorders float adds, so it is not bit-identical to the unsplit kernel;
- masking, causal, ALiBi, GQA and the all-masked-row-is-zero rule follow the unsplit kernel.

**Wiring (0.20.34):** the f32 Vulkan `FlashAttn` dispatch (`vulkan_dispatch::flash_attn_f32`) now calls the split kernel with 8 splits when `decode_splits(b, hq, sq, sk)` says so: `sq == 1`, `b*hq <= 16`, `2048 <= sk <= 4096`. Those thresholds are read off the table below (one device, f32; the 32/64-head cells gained ~1.2x and are left on the unsplit kernel so the bit-pattern change is confined to where the win is clear). It is a fixed prior: no planner/Judge variant yet, no FKC contract, no bf16/f16 variant, so a real bf16/f16 decode does not reach it. `fuel-dispatch/examples/flash_attn_route_check.rs` (live GPU, `b=1 hq=8 hkv=2 sq=1 sk=3000 d=64`) shows the route: the dispatch output is bit-identical to a direct `flash_attn_f32_split_bytes(.., 8)` call, differs from the unsplit kernel in the last bits (max abs 5.6e-9), and is 4.6e-9 from an f64 CPU reference. That proves which kernel ran, not any speedup at model level. **Results in the routed region are NOT bit-identical to the unsplit path** (max abs difference 5.6e-9 against it; 4.6e-9 against an f64 reference), so a consumer comparing exact outputs across versions will see a change. **The routed shape (f32, `sq == 1`, `b*hq <= 16`, `2048 <= sk <= 4096`) is rarely hit by real decode today** (real decode is bf16/f16): no end-to-end claim until a Qwen3 tokens/s measurement exists.

## Correctness

`flash_attn_split_probe check` compares both kernels with an f64 CPU reference on six cases (decode and prefill shapes; GQA 1:1, 4:1, 2:1; causal; ALiBi; `splits` 1, 2, 3, 4, 8, 16; `Sk` 700 to 4096; `D` 64 and 128). Max absolute error: unsplit 2e-8 to 6e-8, split 8e-9 to 6e-8. The assertion bound is 1e-4, so it is loose by about three orders of magnitude; the printed errors are the evidence, and a bug that moved the result by less than 1e-4 would pass. Run it with `--ignored --nocapture`.

## Latency

Median microseconds per call over 60 interleaved rounds (variants round-robin inside each shape, so the box's slow stretches hit all of them), two complete runs, `D = 128`, `sq = 1`, no causal mask. Every number **includes** about 125-275 us of readback (`download_bytes`) and the submit/wait overhead, which a fused decode step would not pay per layer; the scratch allocation of the split path is also inside its number. `-` = chunk would exceed 1024 keys.

| h_q (h_kv) | Sk | unsplit run1 / run2 | split 4 | split 8 | split 16 |
|---|---:|---:|---:|---:|---:|
| 8 (8) | 1024 | 509 / 586 | 436 / 504 | 424 / 515 | 430 / 500 |
| 8 (8) | 2048 | 790 / 740 | 578 / 543 | 611 / 533 | 584 / 531 |
| 8 (8) | 4096 | 1096 / 1034 | 704 / 626 | 663 / 579 | 647 / 547 |
| 32 (8) | 4096 | 1115 / 1173 | 949 / 995 | 901 / 982 | 876 / 942 |
| 64 (8) | 4096 | 1412 / 1499 | 1211 / 1306 | 1139 / 1197 | 1147 / 1217 |
| 64 (8) | 1024 | 724 / 796 | 727 / 782 | 706 / 786 | 721 / 782 |

A third run through the final example binary agrees (`h_q = 8`, `Sk = 4096`: unsplit 1075, split 8 638, split 16 607; `h_q = 64`, `Sk = 4096`: 1518 vs 1271 / 1230). (split 2 and 32 are in the probe output; omitted here. The run-to-run spread of one cell is about +-15%, so only differences larger than that are read.)

Reading, for this box and these shapes only:
- `h_q = 8`, `Sk = 4096`: 1034-1096 us down to 547-647 us, **1.7-1.9x** with 8 or 16 splits, against the hypothesis threshold of 1.5x (stated for `Sk` 4k-32k; only `Sk <= 4096` can be measured because the kernel family is capped there). `Sk = 2048`: 1.3-1.4x. `Sk = 1024`: 1.15-1.2x, close to the spread.
- `h_q = 32`: about 1.25x at `Sk = 4096`; `h_q = 64`: 1.2-1.25x at `Sk = 4096` and no change at `Sk = 1024`. The gain shrinks as heads fill the GPU, and does not turn into a loss in any measured cell. The row's "neutral when heads >= CU count" half holds.
- The per-call floor is about 400 us at `Sk = 1024` for every variant, so in a real decode step (several attention layers per submission, no per-call readback) the same absolute saving, 450-490 us per call at `h_q = 8`, `Sk = 4096`, is a different fraction of the step. **End-to-end decode speed was not measured.**
- Best `splits` was 8 or 16 everywhere it mattered; 4 left some on the table. That is one device, one `D`, one dtype.

## bf16 and f16 (added 0.20.38)

`flash_attn_split_{bf16,f16}.glsl` + `flash_attn_combine_{bf16,f16}.glsl`, driven by `VulkanBackend::flash_attn_{bf16,f16}_split_bytes(.., splits)`. Same structure as the f32 pair; Q/K/V/alibi are read as the half type, the partials in the scratch buffer stay f32, and the combine pass narrows once at the final store (as the unsplit half kernels do: math at f32, narrow on output). The dispatch (`vulkan_dispatch::flash_attn_{bf16,f16}`) now uses the same `decode_splits` rule as f32.

**Correctness** (`flash_attn_split_probe check`, all three dtypes, same six cases as f32, inputs rounded to the dtype first so the f64 reference sees what the GPU reads): every case passes for the unsplit and the split half kernels against a half-ulp relative bound (`2^-8` for bf16, `2^-11` for f16, each +1%) plus 2e-6 absolute. Split-vs-unsplit max abs difference: bf16 up to 1.9e-6, f16 up to 1.5e-5 on outputs of magnitude ~0.01-0.1, i.e. a few outputs differ by one half-ulp, none by more; 0 on the cases where `splits` does not change the add order (prefill, `splits = 1`). **Control:** with the bf16 bound cut to 10% the same run fails (`unsplit Bf16 off its reference by 2.7e-5 past the bound`), so the bound can reject. The route itself is checked by `fuel-dispatch/examples/flash_attn_route_check.rs` for all three dtypes: dispatch output bit-identical to a direct split(8) call, not to the unsplit kernel (max abs difference 5.6e-9 f32, 4.8e-7 bf16, 6.0e-8 f16).

**Latency** (`flash_attn_split_probe latency bf16|f16|f32`, median us over 60 interleaved rounds, ONE run per dtype, `D = 128`, `sq = 1`, includes readback; spread of one cell is about +-15%, so only larger differences are read). Best split of 4-32 against unsplit:

| h_q (h_kv) | Sk | bf16 unsplit / best | f16 unsplit / best | f32 unsplit / best |
|---|---:|---|---|---|
| 8 (8) | 1024 | 435 / 332 | 410 / 321 | 446 / 328 |
| 8 (8) | 2048 | 614 / 367 | 601 / 353 | 704 / 499 |
| 8 (8) | 4096 | 1026 / 578 | 883 / 410 | 1052 / 602 |
| 16 (8) | 1024 | 576 / 498 | 447 / 375 | 597 / 528 |
| 16 (8) | 2048 | 756 / 582 | 584 / 395 | 739 / 582 |
| 16 (8) | 4096 | 1042 / 686 | 890 / 511 | 1085 / 671 |
| 32 (8) | 4096 | 1063 / 904 | 1041 / 885 | 1118 / 858 |
| 64 (8) | 4096 | 1281 / 1084 | 1329 / 1120 | 1362 / 1110 |

Reading, this box and these shapes only: at `b*hq <= 16` and `Sk >= 2048` the half types gain 1.3-1.8x (f16 up to 2.2x at `h_q = 8, Sk = 4096`, 883 to 410 us), consistent with f32; at 32-64 heads about 1.15-1.3x for all three; at `Sk = 1024` 1.15-1.3x, near or inside the spread. f32 at `h_q = 16`, `Sk = 2048` is 1.27x, the thinnest cell the rule routes. `decode_splits` therefore applies one rule to all three dtypes. Qwen3-0.6B's decode has 16 query heads and 8 KV heads, which is the `h_q = 16` row.

**Does a real model reach this route? No, for Qwen3 on Vulkan (established 2026-10-21 at origin/main b950fa81, by reading the model builders and scanning a graph).** (1) `vulkan_dispatch::flash_attn*` rejects a runtime `k_len != sk`, and decode builds a capacity-K KV cache, so a decode `FlashAttn` node would fail to realize on Vulkan; `fuel-dispatch/src/decode_flash.rs` therefore pins its only flash arm to CUDA and emits it only when a CUDA flash kernel is bound and a CUDA device is in the topology. (2) `qwen3_attn_with_kv_writes` (the Qwen3 decode builder) offers no flash arm at all ("no flash-decode arm is offered", because a single `k_len` cannot represent a sliding window), so even CUDA does not get one there. (3) A scan of the Qwen3 prefill graph built by `Qwen3Model::forward_hidden` (2-layer test config, 128 nodes) found 0 `Op::Fused(FLASH_ATTN)` nodes and 0 branch nodes, i.e. the model builds attention decomposed (matmul, scale, mask, softmax, matmul); no optimizer pass in `fuel-graph` creates a `FlashAttn` from that pattern, and no model builder or example in the repo calls the public `Tensor::flash_attn` (its callers are tests and the Judge probe; the CUDA arm emitter builds the node directly). So the split-KV routes shipped in 0.20.32-0.20.38 serve a caller that builds a `FlashAttn` node with `k_len == sk` on Vulkan (none of the Qwen3 builders do); no Qwen3 tokens/s change follows from them, and none is claimed. Not checked: other model builders, and a full Qwen3 decode run on Vulkan counting kernel dispatches (the scan is of the graph, not of dispatches).

## Not measured

- Any model-level tokens/s (none is claimed: see above, Qwen3 on Vulkan does not reach this route); `Sk > 4096` (both wrappers refuse it: the split wrapper keeps the unsplit cap, so the row's 4k-32k range is only measurable at its low end); prefill (`sq > 1`) speed, only its correctness; the AMD iGPU; a planner/Judge-chosen `splits` (the shipped rule is a fixed threshold read off these tables, one device).
- Whether the hand-written SPIR-V in the repo reproduces from the GLSL: the new `.spv` files were built with `glslangValidator -S comp -V`; the committed `flash_attn_f32.spv` is not byte-identical to any compiler invocation tried here (SDK version differences), so these two are not bit-comparable with it either.

## Reproduce

```text
pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release -p fuel-vulkan-backend --example flash_attn_split_probe -- check
pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release -p fuel-vulkan-backend --example flash_attn_split_probe -- latency
```
