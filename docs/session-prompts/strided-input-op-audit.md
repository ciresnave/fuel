# Strided-input op audit (deferred follow-up)

User question raised 2026-06-01 during Phase C any-axis work — captured
here for after the eager-Tensor retirement program completes (Phase D
through H are higher priority).

## The question

> If the main reason for the axis limitation was that existing code
> didn't support strided inputs, what ops that don't currently support
> strided inputs could we modify to support strided inputs? The more
> we can work with data without having to move or change it, the
> faster sequences of kernels (or fused ones) can be.

## Why this matters

`KernelCaps::strided_input()` lets an op consume non-contiguous input
layouts directly (broadcast / transpose / slice views) without the
executor running auto-Contiguize to materialize the bytes first. The
executor's auto-Contiguize pass copies the entire view's bytes into a
fresh contiguous buffer — bandwidth-bound work that's wasted whenever
the next kernel could have walked the original strides.

Some ops are already strided-input-aware (CUDA strided-input sweep
landed ~108 baracuda registrations on 2026-05-24; Vulkan unary/affine/
clamp/powi landed on the same day). Others still force auto-Contiguize.

## Likely candidates for an audit

The high-value targets — ops that frequently sit downstream of a
broadcast / transpose / slice in real workloads:

- **Reductions** — `Sum/Mean/Max/Min/Reduce` family. The kernel walks
  one reduce axis and aggregates; doing that walk over a strided
  input is one extra index calc per element. Big win for batched
  reductions over a sliced cache.
- **Matmul** — already accepts strided inputs in some forms (the
  CUDA `gemm_config` was extended to accept `lda > row_size` for
  the BERT K^T pattern), but the kernel registration matrix might
  not advertise `strided_input` capability uniformly. Worth checking
  which dtypes / backends do and don't.
- **Concat** — variadic-uniform; each input could in principle be a
  strided view. Currently auto-Contiguized.
- **Cast** — pure per-element transform; strided input is trivial.
- **Cmp / Where** — same shape as binary elementwise but currently
  unclear if registered with `strided_input`.
- **Indexing ops** — `IndexSelect`, `Gather`, `IndexAdd`, `ScatterAdd`
  — kernels already do index lookups; adding stride awareness costs
  one stride-vector pass through.
- **Norm family backward** — `RmsNormLastDimBackward`,
  `LayerNormLastDimBackward`. Training-path; could benefit if the
  gradients come from a permuted backward graph.
- **FlashAttn backward** — same family; auto-Contiguize on dK / dV
  may be wasted work.

Less likely (kernel structure isn't stride-friendly):

- **Conv2d** — the im2col / direct-convolution kernels assume
  contiguous NCHW; adding strides would mean the kernel walks them
  for every (n, c, h, w) iteration, which adds significant per-op
  cost.
- **QMatMul** — quant-block dequant assumes contiguous block bytes.
- **FlashAttn forward** — Q/K/V reads are the inner loop; stride
  arithmetic per inner step would be expensive without a tiling
  rewrite.

## Audit method

1. Enumerate every `table.register(...)` call across the four backend
   wrapper files (`baracuda_dispatch.rs`, `vulkan_dispatch.rs`,
   `dispatch.rs` for CPU, `aocl_dispatch.rs` / `mkl_dispatch.rs` for
   vendor CPU).
2. For each entry, check whether it uses `register_with_caps` /
   `register_with_caps_and_precision` and whether the caps include
   `strided_input`. The default is conservative (no strided input).
3. For each `strided_input: false` entry, ask:
   - Does the kernel's inner loop already index by strides?
     (Then flipping the cap is free.)
   - If not, would a simple stride-aware variant of the kernel be
     cheap to write? (Then write it.)
   - If the kernel structure fundamentally assumes contig (conv2d,
     flash_attn forward), leave it and document that auto-Contiguize
     is intrinsic.
4. Test the flipped caps with synthetic strided inputs (broadcast,
   transpose, slice) and compare against the contig-auto-Contiguize
   path for both correctness and perf.

## Why deferred

The Phase A–H eager-Tensor retirement program is the active priority;
this audit doesn't unblock any model port. Worth picking up after
Phase H (eager removed) so the work doesn't duplicate effort on code
that's about to be deleted.

Related memories:
- [[project_cuda_strided_input_sweep]] — 2026-05-24 CUDA flip
- [[project_vulkan_strided_unary_affine_clamp_powi]] — 2026-05-24 Vulkan flip
