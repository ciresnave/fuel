# Port: Conv3D primitive (Qwen3-VL temporal patch embedding)

## Eager source

- `fuel-transformers/src/models/multimodal/qwen3_vl/conv3d_temporal_2.rs` (80 LOC)
  — A Conv3D specialization for temporal-patch-2 (i.e. kernel depth = 2,
    stride = 2 along the temporal axis, kernel/stride 14 along H/W).
    Eager calls into a generic Conv3D backend op.

## Lazy module name

`fuel-core/src/lazy_conv3d.rs` (new file). Exports
`apply_conv3d_temporal_2(input, weight, bias, stride_t, stride_hw)`
returning a `LazyTensor` of shape `(B, out_c, T_out, H_out, W_out)`.

## Architecture summary

Qwen3-VL receives video as `(B, in_c=3, T, H, W)` and applies a
3D patch embedding with kernel = (2, 14, 14) and stride = (2, 14, 14).
Output is `(B, embed_dim, T/2, H/14, W/14)`. The patch sequence is
then flattened to `(B, T/2 * H/14 * W/14, embed_dim)` and fed into
the ViT-style transformer.

**Strategy: decomposition, not a new lazy op.** The lazy graph has
no native Op::Conv3D and adding one is significantly more involved
than the single consumer warrants. Decompose as:

1. Temporal-axis pair: split `T` into chunks of 2 along the time
   axis, producing two tensors of shape `(B, in_c, T/2, H, W)`
   each.
2. Reshape each chunk to `(B, in_c, T/2, H, W)` → treat as
   `(B * T/2, in_c, H, W)` and apply 2D conv with kernel = (14, 14),
   stride = (14, 14), using the corresponding temporal slice of the
   3D weight.
3. Sum the two 2D-conv outputs (this realizes the temporal
   convolution at kernel depth = 2).
4. Reshape back to `(B, out_c, T/2, H_out, W_out)`.

This stays inside the existing Conv2D lazy primitive and matches
the eager behavior bit-for-bit for the (2, 14, 14) case.

## Primitives needed

- None new — uses existing `LazyTensor::conv2d`, `narrow`, `reshape`,
  and binary add. Decomposition is graph-level only.

## Reusable modules

- `lazy_vit` — has the patchify + flatten pattern for the 2D case.
  Use its `apply_patch_embed` as a sibling reference.
- `LazyTensor::conv2d` — existing primitive.

## Open questions

- Qwen3-VL also supports an "image" mode (single frame, T=1).
  Decomposition must handle T=1 without falling apart — confirm by
  reading the eager `forward` path in
  `multimodal/qwen3_vl/vision.rs` and the upstream HF transformers
  reference.
- Are there variants beyond `temporal_2` we need to support? The
  eager file is named `conv3d_temporal_2.rs`, suggesting the design
  is specialized. Check the Qwen3-VL config — if `temporal_patch_size`
  is configurable, generalize the decomposition to arbitrary
  kernel depth (slice and weighted-sum N times instead of 2).

## Splits

Single session unless investigation surfaces additional variants:

1. Sub-port 1 (this spec): `apply_conv3d_temporal_2` decomposition
   + unit tests against a hand-computed expected output.
2. Sub-port 2 (only if Qwen3-VL config requires it): generalize to
   `apply_conv3d_temporal_n(n: usize, ...)`.

## Test strategy

- Tiny config: in_c=3, out_c=4, T=4, H=28, W=28, patch=(2,14,14).
  Build the same weight tensor for both an eager Conv3D reference
  (via `tensor::Tensor::conv3d`) and the lazy decomposition; assert
  they match within 1e-6.
- Edge case: T=2 (smallest non-trivial). T=1 (image mode) — should
  either error gracefully or produce the same answer as a 2D conv
  with the first temporal slice of the weight.

## References

- Eager source: `fuel-transformers/src/models/multimodal/qwen3_vl/conv3d_temporal_2.rs`
- Qwen3-VL paper: <https://arxiv.org/abs/2411.10440>
- HF transformers reference:
  `transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`
  (look at `Qwen3VLVisionPatchEmbed`).
- Sibling: `lazy_vit::apply_patch_embed` (2D analogue).
