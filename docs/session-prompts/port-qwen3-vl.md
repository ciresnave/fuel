# Port: Qwen3-VL (text + vision + composition)

## Eager source

- `fuel-transformers/src/models/multimodal/qwen3_vl/text.rs` (403 LOC)
  — Qwen3-VL language model (Qwen3-style decoder with VL-specific
    embed adapter).
- `fuel-transformers/src/models/multimodal/qwen3_vl/vision.rs` (588 LOC)
  — Vision tower: Conv3D patch embed → ViT-style transformer with
    DeepStack residual injection at intermediate layers + cu_seqlens
    variable-length attention.
- `fuel-transformers/src/models/multimodal/qwen3_vl/conv3d_temporal_2.rs` (80 LOC)
  — Patch-embed helper. **Ported separately** in
    [port-conv3d.md](port-conv3d.md) — do that first.
- `fuel-transformers/src/models/multimodal/qwen3_vl/config.rs` (74 LOC)
  — `Qwen3VLConfig` HF deserialization.
- `fuel-transformers/src/models/multimodal/qwen3_vl/mod.rs` (273 LOC)
  — Top-level composition: image/video → vision encoder → embed
    projector → text LM with image-token slot injection.

## Lazy module name

- `fuel-core/src/lazy_qwen3_vl_text.rs`
- `fuel-core/src/lazy_qwen3_vl_vision.rs`
- `fuel-core/src/lazy_qwen3_vl.rs` (composition + config)

Mirrors the eager file split so it's easy to navigate.

## Architecture summary

Vision tower:
1. **Conv3D temporal patch embed** (kernel=(2,14,14), stride=(2,14,14))
   — see port-conv3d.md.
2. **Patch sequence** → ViT-style transformer.
3. **Window attention with cu_seqlens** — within each "image" the
   patches form independent sequences. cu_seqlens carries the
   cumulative sequence lengths so the attention kernel only
   attends within an image, not across images in a packed batch.
4. **DeepStack residual injection** — at specific layers
   (configurable list), the vision intermediate is projected and
   added into the corresponding text-side embedding slot. This
   tightens vision↔text coupling beyond the usual final-layer
   projection.
5. **MROPE (multi-axis RoPE)** — vision positions are 2D (h, w),
   text positions are 1D (t), so the rotary table is built from
   three independent axes concatenated along the head dim. Eager
   has `mrope_section` config for axis-dim split.

Composition (`mod.rs`):
- `Qwen3VLModel::forward(image_pixels, video_pixels, text_tokens,
  start_pos)` → logits.
- Image/video tokens are placeholder ids in `text_tokens`; the
  vision encoder produces embeddings that get scatter-written into
  the corresponding positions of the text embed sequence before the
  LM forward pass.

Text LM:
- Qwen3-style decoder (already shipped as `lazy_qwen3`) with two
  changes:
  1. RoPE switched to MROPE.
  2. `forward_embeds_with_mrope_positions(embeds, pos_grid)`
     accepts the per-token MROPE position tuple instead of a single
     start_pos scalar.

## Primitives needed

- **Conv3D temporal patch embed** — port-conv3d.md.
- **cu_seqlens variable-length attention** — needs a path that
  builds an attention mask from cumulative lengths and feeds it
  into `LazyTensor::sdpa_*`. Two options:
  1. Materialize a block-diagonal attention mask host-side, pass
     as a tensor. Simple, but O(L^2) mask memory.
  2. Pack into a packed-sequence representation that the SDPA op
     already handles. Check whether the existing `paged_attn`
     surface handles this — if cu_seqlens semantics match
     paged_attn's per-sequence context_lens, route through that.
  Decision deferred to the implementation session; pick whichever
  reads cleaner against the existing graph ops.
- **MROPE table** — three independent RoPE tables concatenated
  along head dim. Host-built `const_f32_like`, same pattern as
  Llama-3 RoPE.
- **Scatter-write into text-embed sequence** at image-token
  positions. `LazyTensor::index_put` or `scatter_add` should
  suffice — verify the existing surface supports this.

## Reusable modules

- `lazy_qwen3` — text LM forward. Will need extension or wrapping
  for MROPE + `forward_embeds_with_mrope_positions`.
- `lazy_vit` — ViT transformer pattern.
- `lazy_paligemma`, `lazy_llava` — composition pattern with image
  embed injection.
- `lazy_conv3d::apply_conv3d_temporal_2` (after port-conv3d.md
  ships).

## Open questions

- Conv3D variant: does Qwen3-VL config expose a configurable
  `temporal_patch_size` or is it always 2? Drives whether
  port-conv3d.md needs the generalization split.
- cu_seqlens vs paged_attn — concrete: does `paged_attn` work for
  the vision-only forward pass (no KV cache, no incremental
  decode)? If not, we need the block-diagonal-mask path.
- DeepStack injection points: hard-coded layer indices or config?
  HF Qwen3-VL `vision_config.deepstack_visual_indexes` is a list.
  Carry it as config.
- Image preprocessing (resize, normalize, patch-pad to mod-14): is
  it host-side caller responsibility, or do we need a lazy
  preprocess helper? Check the eager `mod.rs` forward path —
  if it does the resize itself, lift it into the lazy port too.

## Splits

This port is large (~1.4k LOC eager). Recommended split:

1. **Sub-port 1**: `lazy_qwen3_vl_vision` with Conv3D patch embed
   + ViT transformer + cu_seqlens attention + DeepStack hooks.
   Standalone tests against a tiny image-only config.
2. **Sub-port 2**: `lazy_qwen3_vl_text` — Qwen3 + MROPE +
   `forward_embeds_with_mrope_positions`. Standalone tests
   against a tiny tokens-only config.
3. **Sub-port 3**: composition `lazy_qwen3_vl` — image+text
   end-to-end forward, DeepStack residual wiring, integration
   test that prefixes a `(1,3,4,28,28)` video to a short prompt.

Sub-ports 1 and 2 can ship in either order; sub-port 3 gates on
both.

## Test strategy

- Tiny vision config: T=4 frames, H=W=28, embed_dim=16, depth=2,
  num_heads=4, patch=(2,14,14). Verify (B, num_patches, embed_dim)
  output shape and finite values.
- Tiny text config: vocab=32, hidden=16, layers=2, heads=4, kv=2.
  Verify MROPE position grid drives different attention than
  scalar start_pos.
- DeepStack: zero out the visual residual, confirm the text
  forward path produces the same logits as `lazy_qwen3` baseline
  (regression sanity).
- Composition end-to-end: build the tiny vision + text + composer,
  feed `(1,3,4,28,28)` image + 8-token text with 2 image-token
  slots, assert logits shape `(1, 8, vocab)` and finite.

## References

- Eager source: `fuel-transformers/src/models/multimodal/qwen3_vl/*`
- Qwen3-VL paper / blog: <https://qwenlm.github.io/blog/qwen3-vl/>
- HF transformers reference:
  `transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- Already-shipped composition patterns: `lazy_llava`, `lazy_paligemma`.
- Already-shipped MROPE-adjacent: `lazy_glm4` (interleaved 2-axis
  RoPE).
