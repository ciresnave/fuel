# Split-KV decode attention on Vulkan: what was measured

ROADMAP item 17, row **H12** (FlashAttention `num_splits_heuristic`). Measured 2026-10-20, RTX 4070 Laptop GPU via Vulkan (`PreferDiscrete`), release build, `scripts/gpu-run.ps1`, nothing else running. Probe: `fuel-vulkan-backend/examples/flash_attn_split_probe.rs` (an example, not an `#[ignore]` test, to keep the ignore ceiling).

## What the production kernel does

The Vulkan attention forward that runs today is `fuel-kernels-source/kernels/flash_attn_f32.glsl` (not the Slang file the harvest row cited): one workgroup per `(b, h_q, q_i)`, scores for the whole key range in shared memory (`Sk <= 4096`), and a V-accumulate loop that walks every key serially per output element. A decode step (`sq = 1`) therefore launches `b * h_q` workgroups however long the cache is.

## What was added

`flash_attn_split_f32.glsl` + `flash_attn_combine_f32.glsl`, driven by `VulkanBackend::flash_attn_f32_split_bytes(.., splits)`:
- pass 1 runs `(splits, 1, b*h_q*sq)` workgroups, each over a chunk of `ceil(Sk / splits)` keys (at most 1024), writing an unnormalised partial `o[d]`, the chunk max `m` and sum `l`;
- pass 2 merges the partials of a row in fixed split order: `M = max m`, `L = sum l*exp(m-M)`, `out = sum o*exp(m-M) / L`. Deterministic for a given `splits`; it reorders float adds, so it is not bit-identical to the unsplit kernel;
- masking, causal, ALiBi, GQA and the all-masked-row-is-zero rule follow the unsplit kernel.

**Wiring (0.20.34):** the f32 Vulkan `FlashAttn` dispatch (`vulkan_dispatch::flash_attn_f32`) now calls the split kernel with 8 splits when `decode_splits(b, hq, sq, sk)` says so: `sq == 1`, `b*hq <= 16`, `2048 <= sk <= 4096`. Those thresholds are read off the table below (one device, f32; the 32/64-head cells gained ~1.2x and are left on the unsplit kernel so the bit-pattern change is confined to where the win is clear). It is a fixed prior: no planner/Judge variant yet, no FKC contract, no bf16/f16 variant, so a real bf16/f16 decode does not reach it. `fuel-dispatch/examples/flash_attn_route_check.rs` (live GPU, `b=1 hq=8 hkv=2 sq=1 sk=3000 d=64`) shows the route: the dispatch output is bit-identical to a direct `flash_attn_f32_split_bytes(.., 8)` call, differs from the unsplit kernel in the last bits (max abs 5.6e-9), and is 4.6e-9 from an f64 CPU reference. That proves which kernel ran, not any speedup at model level.

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

## Not measured

- Any model-level tokens/s; `Sk > 4096` (both wrappers refuse it: the split wrapper keeps the unsplit cap, so the row's 4k-32k range is only measurable at its low end); bf16/f16; prefill (`sq > 1`) speed, only its correctness; the AMD iGPU; a `splits` heuristic (the row asks for one; the data here only says 8-16 on an RTX 4070 at `h_q <= 8`).
- Whether the hand-written SPIR-V in the repo reproduces from the GLSL: the new `.spv` files were built with `glslangValidator -S comp -V`; the committed `flash_attn_f32.spv` is not byte-identical to any compiler invocation tried here (SDK version differences), so these two are not bit-comparable with it either.

## Reproduce

```text
pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release -p fuel-vulkan-backend --example flash_attn_split_probe -- check
pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release -p fuel-vulkan-backend --example flash_attn_split_probe -- latency
```
