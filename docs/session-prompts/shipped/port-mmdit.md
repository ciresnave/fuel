# Port: MMDiT (SD3 / Flux foundation)

## Eager source

- `fuel-transformers/src/models/diffusion/mmdit/blocks.rs` (522 LOC)
  — DoubleStreamBlock + SingleStreamBlock + modulated attention.
- `fuel-transformers/src/models/diffusion/mmdit/embedding.rs` (209 LOC)
  — Sinusoidal timestep embed + label embed + RoPE patch positions.
- `fuel-transformers/src/models/diffusion/mmdit/model.rs` (256 LOC)
  — Top-level MMDiT model + forward.
- `fuel-transformers/src/models/diffusion/mmdit/projections.rs` (131 LOC)
  — Q/K/V projections + AdaLN modulation projection helpers.

## Lazy module name

`fuel-core/src/lazy_mmdit.rs` (single file — sub-module organization
in source via `mod blocks; mod embedding; mod model; mod projections;`
inside the file is fine and matches existing
`lazy_*` module patterns).

Actually: matches the eager split better as four files.
`fuel-core/src/lazy_mmdit/{mod.rs, blocks.rs, embedding.rs,
projections.rs}` — first lazy_* port that warrants a directory
module rather than a single file.

## Architecture summary

MMDiT (Multimodal Diffusion Transformer) is the SD3 / Flux
substrate: a transformer that processes **joint** text and image
token streams with shared (or partially shared) attention.

Two block types:

1. **DoubleStreamBlock** — text and image have separate Q/K/V
   projections + separate AdaLN modulation params, but attention
   keys/values are concatenated across modalities so each token
   attends to both modalities. Output is split back to per-modality.
2. **SingleStreamBlock** — text and image are concatenated, run
   through a unified attention + MLP (after some layers of
   DoubleStream, the streams are joined).

Modulation: every block reads a `(B, n_mod_params * dim)` vector
from a timestep + label embedding, used as scale/shift for AdaLN
and as an output gate. SD3 has 6 modulation params per stream per
block; Flux has 6 or 9 depending on block type.

## Primitives needed

- **AdaLN modulation** — `(scale, shift, gate) = chunk(mod_params, 3)`
  applied as `(1 + scale) * norm(x) + shift` then `gate * out`.
  Standard primitive; reusable as `lazy_mmdit::apply_modulation`.
- **2D RoPE for patch positions** — image patches have (h, w)
  positions; build a 2-axis RoPE table host-side.
- **Concat-attention** — concat text and image K/V along sequence
  axis before SDPA. Just `LazyTensor::cat`.

## Reusable modules

- `lazy_sd_unet` — has timestep embed + AdaLN-adjacent patterns
  (though sd_unet is UNet, not DiT). Reference for the embedding
  helpers.
- `lazy_vit` — patch embed.
- `lazy_clip` — text encoder upstream of MMDiT (SD3 uses CLIP +
  T5 text encoders).

## Open questions

- SD3 vs Flux MMDiT — they have slight architectural differences
  (Flux has a "qk_norm" applied, SD3 doesn't; Flux has a parallel
  attention shape; etc.). Is the eager file a SD3-only MMDiT, a
  Flux-only one, or a unified config-driven one? Read `model.rs`
  and `blocks.rs` to confirm.
  - If config-driven: handle both variants.
  - If SD3-only: Flux's MMDiT-like blocks live inside
    `diffusion/flux/model.rs` and port-flux.md picks them up
    separately.
- DoubleStream block count vs SingleStream block count — config?
  Or hard-coded per model?

## Splits

This is the largest substrate for Phase F. Split:

1. **Sub-port 1**: `lazy_mmdit/embedding.rs` + `lazy_mmdit/projections.rs`
   — small standalone primitives. Unit tests for sinusoidal embed,
   2D RoPE.
2. **Sub-port 2**: `lazy_mmdit/blocks.rs` — DoubleStreamBlock +
   SingleStreamBlock + AdaLN modulation. Tested standalone on a
   tiny config.
3. **Sub-port 3**: `lazy_mmdit/mod.rs` (model top + forward) +
   integration test.

## Test strategy

- Tiny config: text_dim=16, image_dim=16, num_heads=4, depth=2
  (1 DoubleStream + 1 SingleStream), seq_text=8, seq_image=16
  (4x4 patches).
- Verify modulation: zero-out scale → output equals plain
  layernorm(x); zero-out gate → output equals residual.
- Verify joint attention: text Q attends to image K → check
  attention weight shape `(B, num_heads, seq_text + seq_image,
  seq_text + seq_image)` in the double-stream concat path.

## References

- Eager source: `fuel-transformers/src/models/diffusion/mmdit/*`
- SD3 paper: <https://arxiv.org/abs/2403.03206> (Scaling Rectified
  Flow Transformers for High-Resolution Image Synthesis).
- Flux release notes: <https://blackforestlabs.ai/announcing-black-forest-labs/>
- Reference impl: <https://github.com/Stability-AI/sd3-ref> and
  <https://github.com/black-forest-labs/flux>.
- Sibling ports: `lazy_sd_unet`, `lazy_vit`, `lazy_clip`.
