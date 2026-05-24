# MoE GEMM design analysis (2026-05-25)

Companion to `fuel-cuda-kernels-retirement-audit.md` (phase 4 needs
this resolution). Captures what Fuel's fused MoE kernels actually do,
what alternatives look like, and the question to put to the baracuda
team.

## What we have today

**FFI surface** (`fuel-cuda-kernels/src/ffi.rs`): three CUDA kernels.

| Kernel | Math | Dtypes | Decode/Prefill |
|---|---|---|---|
| `moe_gemm_wmma` | `Y[t, n] = Σ_k X[t, k] · W[expert(t), n, k]` (FP, non-quant) | half / bf16 | both (different WMMA tile shapes) |
| `moe_gemm_gguf` | Same math with GGUF-quantized W (decode-shape; M small) | Q2K/Q3K/Q4K/Q5K/Q6K/Q8_0 weights × f32 input → f32 output | decode |
| `moe_gemm_gguf_prefill` | Same math, prefill-shape (M large) | Q* weights × f16/bf16 input → output | prefill |

**Consumer surface**: a single Fuel-level user.

- `fuel-nn::moe::moe_gemm` + `moe_gemm_gguf` — public wrappers binding
  the FFI.
- `fuel-transformers::fused_moe::FusedMoe` — the consumer; calls
  `moe::moe_gemm_gguf` three times per layer (gate / up / down).
- Models using it: `qwen3_moe` (non-quant), `quantized_qwen3_moe`
  (GGUF). Production MoE models in fuel-transformers.

## What the fused kernel actually does

(Reading `moe_wmma.cu`'s `moe_gemm_grouped_kernel`.)

For a forward pass with `M` tokens, `N` experts (typically 8), `topk`
experts-per-token (typically 2):

1. Caller has pre-sorted tokens by expert via `sorted_token_ids[]` and
   computed per-expert segment offsets via `expert_offsets[]`.
2. Single CUDA launch with `grid = (num_experts, ceil(N/N_BLK), 1)`,
   `block = (128, 1, 1)`.
3. Each block processes `(expert_id, n_tile_idx)`:
   - Walks its expert's token segment (`expert_offsets[i]` →
     `expert_offsets[i+1]`).
   - Tiles the M dim by `M_BLK = 32`, K dim by `K_BLK = 16`.
   - For each `(m_base, k_base)` tile:
     - Cooperative load of A tile (input tokens for this segment) and
       B tile (expert's weight slice) into shared memory.
     - WMMA fragment loads + `mma_sync` (tensor cores; fp32
       accumulator, half/bf16 operands).
   - Stores accumulated C tile back through `sorted_token_ids[]`,
     applying `topk_weights[token]` for top-k > 1 weighting.
4. Decode vs prefill: different WMMA tile shapes
   (`(M=8, N=32, WARPS_N=1)` for decode; `(M=16, N=16, WARPS_N=2)` for
   prefill — smaller M for decode where token count is small).

Quantized variants (`moe_gemm_gguf` / `_prefill`) follow the same
"single launch per gate/up/down, grouped by expert with sorted-token
routing in-kernel" pattern, just substituting block-dequant-then-MV
math for the WMMA path.

## What naive MMVQ × N would lose

If we delete the fused kernels and rewrite `FusedMoe` to use
baracuda's per-expert MMVQ × N + token routing, we lose:

1. **Launch amortization.** Today: 3 launches per MoE layer (gate +
   up + down). Naive MMVQ × N: `3N` launches. For Mixtral-8x7B with
   N=8 experts × 32 layers = 768 launches per token vs. 96 today.
   Each launch has driver overhead (~1-5 µs on a hot stream); for
   decode-time inference where total work per token is ~10 ms, the
   launch tax matters.
2. **Tensor-core utilization for the FP path.** `moe_gemm_wmma`
   explicitly uses WMMA for the non-quant case. Baracuda's MMVQ is a
   matrix-vector kernel (not a tensor-core path). For the FP
   non-quantized MoE path (`moe_gemm_wmma`, not the GGUF variants),
   this is a real perf gap.
3. **In-kernel routing.** Today the kernel reads `sorted_token_ids[]`
   inside the inner loop. With MMVQ × N, the gather (route tokens to
   experts) and scatter (collect results back) become separate
   passes — extra DRAM round-trips and either a CUB-style segmented
   reduction or host-side routing.

## What naive MMVQ × N would gain

1. **Code simplicity.** Drop three .cu files + one .cuh + ~500 LOC of
   moe_utils. Rewrite is on the Fuel side (Rust); leverages
   well-tested MMVQ kernels.
2. **Per-expert kernel selection independent of MoE pipeline.** If
   baracuda ships a better MMVQ for a specific quant format (e.g.
   alpha.35 adds an FP8 MMVQ), MoE picks it up for free.
3. **Trivial extension to new quant formats.** New format = new MMVQ;
   no MoE-specific kernel work.

## Hybrid option: batched MMVQ

The interesting middle path: ask baracuda for a "batched MMVQ" API.

```c
// Sketch — single launch handles all N experts.
fn mmvq_batched_<fmt>_run(
    n_experts: i32,
    n_rows_per_expert: i32,     // output features (N) per expert
    n_cols: i32,                // K
    weights: *const c_void,     // [N_experts, N_rows, K] block-packed
    activations: *const c_void, // [M_tokens, K]
    sorted_token_ids: *const i32, // [M_total]  (M_total = M_tokens × topk)
    expert_offsets: *const i32, // [N_experts + 1] segment offsets
    topk_weights: *const f32,   // [M_total] optional
    output: *mut c_void,        // [M_tokens, N_rows]
    workspace: *mut c_void, workspace_bytes: usize,
    stream: *mut c_void,
) -> i32;
```

This gives us the launch amortization without baracuda absorbing
Fuel's WMMA grouped kernel — they keep MMVQ semantics, just add the
expert-routing dimension. For the FP non-quant case we'd still want
a separate path (or accept the tensor-core loss).

## Recommendation

Frame the baracuda ask as a three-way choice with our preference
ordered:

1. **(Preferred) Hybrid: batched MMVQ × N-experts kernel.** Keeps
   baracuda's MMVQ-centric architecture; adds the routing dim that
   Fuel needs. Single launch per (gate/up/down). Loses tensor-core
   utilization for FP MoE (acceptable — quantized MoE is the
   dominant inference path; FP MoE training is the loss but baracuda
   already doesn't optimize for that).
2. **(Acceptable) Accept Fuel's fused MoE kernels upstream.**
   `moe_gemm_wmma` + `moe_gemm_gguf` + `moe_gemm_gguf_prefill` move
   to baracuda; Fuel's `moe/` subdir deletes. Baracuda gets a
   maintenance burden (Fuel-specific MoE optimizations) but the
   architectural seam stays clean: Fuel imports kernels from baracuda,
   period.
3. **(Fallback) Decline both; we rewrite to MMVQ × N + routing on
   Fuel side.** Multi-week rewrite of `fuel-transformers::fused_moe`.
   Loses launch amortization (~8× launches for Mixtral-class models)
   and tensor-core utilization on FP MoE path. Acceptable if the
   inference workload doesn't have MoE in the hot path.

The hybrid option is the architecturally cleanest answer if baracuda
is willing to take it on — it's a natural extension of their existing
MMVQ surface (they already have per-block-format MMVQ kernels; adding
a routing dimension is a kernel-template change, not a new family).

## Open numbers (would refine the ask)

We don't have profiling data on hand to quantify the launch-amortization
cost. The decision arguments above use heuristics (driver launch
overhead, MoE layer count). If the baracuda team wants empirical
numbers before committing to option 1, that's a measurement project
(probably nsight-systems on a Qwen3-MoE decode pass). Flagging here
so we can add the measurement if asked.
