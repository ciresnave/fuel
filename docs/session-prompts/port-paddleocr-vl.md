# Port: PaddleOCR-VL (text + vision + composition)

## Eager source

- `fuel-transformers/src/models/multimodal/paddleocr_vl/text.rs` (1254 LOC)
  — Ernie-style text LM. Decoder-only, RMSNorm, RoPE,
    SwiGLU. Differences from llama/mistral noted below.
- `fuel-transformers/src/models/multimodal/paddleocr_vl/vision.rs` (1222 LOC)
  — OCR-specific ViT with high-resolution multi-aspect-ratio
    handling: image is split into tiles, each tile runs through
    the vision tower, then results are stitched back.
- `fuel-transformers/src/models/multimodal/paddleocr_vl/config.rs` (398 LOC)
  — Config types for both stacks + composition.
- `fuel-transformers/src/models/multimodal/paddleocr_vl/mod.rs` (1109 LOC)
  — Top-level composition + image-tile preprocessing + image-token
    slot management in the text stream.

## Lazy module name

- `fuel-core/src/lazy_paddleocr_vl_text.rs`
- `fuel-core/src/lazy_paddleocr_vl_vision.rs`
- `fuel-core/src/lazy_paddleocr_vl.rs` (composition + config)

## Architecture summary

OCR-VL is a vision-language model specialized for document
understanding. Key differences from the typical multimodal
template:

1. **High-resolution multi-tile vision**: input image is split
   into overlapping or non-overlapping tiles (e.g. 4 corner tiles
   + 1 thumbnail). Each tile runs the ViT independently. Tile
   embeddings are concatenated with positional markers indicating
   which tile each came from.
2. **Aspect-ratio-aware partitioning**: the tile grid adapts to
   image aspect ratio (e.g. landscape → 2x1, portrait → 1x2,
   square → 1x1). Eager has a partitioning helper.
3. **Ernie-style text LM**: Baidu's Ernie variant differs from
   llama in:
   - Embedding scaling factor `embedding_multiplier` baked at load
     (skip if 1.0).
   - LayerNorm vs RMSNorm — config-driven.
   - GQA with possibly non-standard kv ordering — verify against
     the safetensors weight layout.

## Primitives needed

- **Tile partitioning** — host-side image-to-tiles helper. Pure
  Vec<f32> / image-crate work, no graph ops.
- **Aspect-ratio chooser** — host-side function from
  (H, W) to (rows, cols).
- Standard ViT primitives — all exist.

## Reusable modules

- `lazy_vit` — ViT backbone.
- `lazy_pixtral` — Pixtral has high-resolution multi-image
  partitioning patterns; cross-reference for the tile-concat layout.
- `lazy_qwen3_vl` (after that ships) — image-token slot scatter.
- `lazy_clip` — image preprocessing normalization helpers.

## Open questions

- Exact tile partitioning algorithm — PaddleOCR-VL's choice differs
  from Pixtral and LLaVA-Next. Read the eager `mod.rs`
  `partition_image` (or equivalent name) carefully.
- Is the text LM literally Ernie, or PaddleOCR's own variant? Walk
  the eager `text.rs` and identify each deviation from a stock
  decoder. List them in this spec before starting the port (fill
  the deviations into a sub-section here once known).
- DeepStack or final-layer-only injection? Read eager `mod.rs`.

## Splits

Largest non-diffusion port. Recommended split:

1. **Sub-port 1**: `lazy_paddleocr_vl_text` — Ernie-style text LM
   standalone. Forward + `forward_embeds`.
2. **Sub-port 2**: `lazy_paddleocr_vl_vision` — tile-aware ViT.
   Standalone tiny test on a fake multi-tile image.
3. **Sub-port 3**: composition + aspect-ratio chooser + tile
   partitioning + image-token slot scatter. End-to-end test.

Each sub-port own commit, own session.

## Test strategy

- Tile partitioning: golden tests for several (H, W) inputs
  matching eager output exactly.
- Vision tower: tiny config, single-tile and 2x2-tile inputs,
  verify output sequence length scales linearly with tile count.
- Text LM: standard decoder smoke test (vocab=32, hidden=16,
  layers=2, heads=4, kv=2).
- Composition: tiny end-to-end, assert logits finite, image-token
  slots correctly replaced by visual embeddings.

## References

- Eager source: `fuel-transformers/src/models/multimodal/paddleocr_vl/*`
- PaddleOCR docs: <https://github.com/PaddlePaddle/PaddleOCR>
- Ernie reference: HF `ernie` modeling files for the text-stack
  deviations.
- Already-shipped: `lazy_pixtral` (multi-tile vision-language),
  `lazy_qwen3_vl` (after it ships — image-token slot pattern).
