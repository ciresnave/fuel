# Session prompt — Complete the lazy PaddleOCR-VL port (NaViT + bilinear preprocessor + binary revival)

## What this session is for

PaddleOCR-VL is one of the binaries quarantined in commit `cfcb35cf`
("Phase H — eager fuel-transformers/models retired"). The README at
`fuel-examples/examples/_paddleocr-vl_retired/README.md` survived; the
`main.rs` was deleted. The reason for quarantine: the lazy v1 port
shipped two deliberate simplifications that aren't acceptable for a
real OCR run:

1. **No dynamic-resolution NaViT.** `fuel-core::lazy_paddleocr_vl_vision.rs:39`
   carries the TODO `Dynamic-resolution NaViT deferred`. The lazy v1
   processes a fixed-tile grid (`(image_size, image_size)` per tile,
   eager-style aspect-ratio chooser + nearest-neighbor host resize per
   tile). The published checkpoint expects the NaViT path — pad each
   image to a multiple of `patch_size`, run the encoder over the full
   variable-resolution patch grid, interpolate the base 27×27 position
   embedding to match. Without it, the encoder's positional signal
   doesn't match what the trained weights expect, so OCR accuracy
   collapses.
2. **No bilinear image preprocessor.** `fuel-core::lazy_paddleocr_vl.rs:35`
   carries the comment `Bilinear image preprocessor deferred`. The
   lazy v1 uses nearest-neighbor host resize inside `forward_with_image`;
   the eager preprocessor (and the retired-binary's `smart_resize` +
   `image::imageops::resize` with `CatmullRom`) is bilinear-style
   with ImageNet-mean/std normalization. Nearest-neighbor at
   document resolutions visibly destroys glyph edges.

The base `lazy_paddleocr_vl_text` (ERNIE decoder with M-RoPE) and the
fixed-tile-grid `lazy_paddleocr_vl_vision` + `lazy_paddleocr_vl`
composition all exist, work, and have unit tests. This session adds
the two missing pieces, then rewrites the binary against the
completed lazy modules.

Estimated total scope: **2-3 focused sessions** (1 for NaViT, 1 for
preprocessor, 0.5 for the binary — split or pack at the
implementer's preference).

## Read first (in this order)

1. **`fuel-transformers/src/_models_retired/multimodal/paddleocr_vl/vision.rs`**
   (1222 LOC) — the eager NaViT reference. Lines 88-250 contain
   `PatchEmbedding` + `interpolate_pos_encoding` (the bilinear position-
   embedding interpolator with LFU cache). Lines 252-297 contain the
   2D RoPE used by the vision attention. Lines 299-387 contain
   `chunked_attention` (online-softmax tiled attention for very
   long sequences — relevant when NaViT processes 14K+ patches from
   high-res documents). Lines 569-657 contain `VisionBlock`. Lines
   904-972 contain `rot_pos_emb` + `build_cu_seqlens` — the variable-
   length attention machinery.
2. **`fuel-transformers/src/_models_retired/multimodal/paddleocr_vl/mod.rs`**
   (1109 LOC) — the eager top-level composition. The image-encoding
   call surface (`encode_image`, `encode_images_separate`,
   `encode_images_multi`, `forward_video`) reveals how the eager
   binary passed `grid_thw` to the encoder. The bilinear preprocessor
   is **not** in this file (the eager pipeline relies on its caller
   for preprocessing — see the deleted binary diff in commit
   `cfcb35cf` for `smart_resize` + the `CatmullRom` resize + ImageNet
   normalization step).
3. **The deleted binary** — `git show cfcb35cf -- fuel-examples/examples/paddleocr-vl/main.rs`
   recovers the 457-line eager binary. The `smart_resize` function at
   lines 209-260 and the `load_image_lazy` helper at lines 264-307
   are the preprocessor spec; the deletion was deliberate (the lazy
   v1 binary held only a single-image smoke test), but the
   preprocessor logic is canonical and should be ported as-is.
4. **`fuel-core/src/lazy_paddleocr_vl_vision.rs`** (~1066 LOC) — the
   shipped fixed-tile-grid encoder. Note the host-side
   `aspect_ratio_chooser` + `partition_image` helpers; both should
   be retained for the legacy fixed-tile path, with the new dynamic
   NaViT path living alongside. The TODO at line 39 documents the
   deferral.
5. **`fuel-core/src/lazy_paddleocr_vl.rs`** (~858 LOC) — the shipped
   top-level composition. `forward_with_image` (line 132) is the
   host-side preprocessing hook; the new bilinear preprocessor
   replaces the nearest-neighbor resize at lines 182-192. The
   `resize_nearest_chw` function at line 300 is what's being
   replaced.
6. **`fuel-examples/examples/_paddleocr-vl_retired/README.md`** —
   the user-facing CLI contract that the revived binary must match.
   Multi-image, batch, and video modes are documented; the lazy v1
   binary punted on all three. This session can scope to single-image
   (matches the retired-binary v1 state) and queue multi-image /
   batch / video as follow-up.

## Preconditions — verify before starting

1. **lazy_paddleocr_vl_text builds and tests pass.** Run
   `cargo test -p fuel-core lazy_paddleocr_vl_text -- --nocapture`
   and confirm green. If broken (eager retirement may have churned
   shared helpers), fix first.
2. **lazy_paddleocr_vl_vision tests pass** under the existing
   fixed-tile-grid path. The NaViT addition should be a new
   forward entry point alongside the existing `forward(...)`,
   not a rewrite — the fixed-tile path stays available for the
   small-image case.
3. **`LazyTensor` has `conv2d` + `layer_norm_affine` + `slice` +
   `concat` + `permute` working on F32 CPU.** All four are exercised
   by the existing lazy_paddleocr_vl_vision encoder; if any have
   regressed, surface before proceeding.
4. **No active churn on `fuel-core/src/lazy_paddleocr_vl*.rs`.**
   Check `git log --since="7 days" fuel-core/src/lazy_paddleocr_vl*`
   for parallel work; coordinate if anyone else is touching those
   files.
5. **`image` crate is already a dev-dep or workspace dep** of
   fuel-examples — verify in `fuel-examples/Cargo.toml`. The
   bilinear preprocessor wants `image::imageops::resize` with
   `FilterType::CatmullRom`. If the crate isn't already available,
   add it to fuel-examples (NOT fuel-core — the preprocessor's
   bilinear math should live in fuel-core but the `DynamicImage`
   loading should stay in the binary; see scope below).

## Scope

### Part 1 — Dynamic-resolution NaViT port (~1 session)

**Goal:** add `PaddleOcrVlNaVitModel` + `PaddleOcrVlNaVitConfig` +
`PaddleOcrVlNaVitWeights` to `fuel-core/src/lazy_paddleocr_vl_vision.rs`
implementing the NaViT forward path. Keep the existing fixed-tile-grid
`PaddleOcrVlVisionModel` unchanged — it's the simpler path for square
images and the test surface for the shared helpers (RoPE tables,
projector, layer norms).

**Architecture:**

NaViT (Native Resolution ViT) processes a single image at its natural
aspect ratio. The image is padded host-side to be `(H', W')` where
`H'` and `W'` are both multiples of `patch_size`. The patch embedding
projection (a `Conv2d` with `stride == patch_size`) produces
`(H'/patch_size, W'/patch_size)` patches. The 2D RoPE tables and the
1D position embedding are recomputed/interpolated for that specific
patch grid.

Concretely:

1. **Config + Weights structs.** `PaddleOcrVlNaVitConfig` mirrors
   `PaddleOcrVlVisionConfig` but drops the fixed `image_size` field
   (NaViT doesn't have one — `patch_size` is the only spatial
   constraint). Add a new `max_pixels` field for the smart-resize
   cap. `PaddleOcrVlNaVitWeights` adds a `base_position_embedding`
   field: the 27×27 grid (= `(image_size_at_train_time/patch_size)^2`)
   that the eager code interpolates from. The existing
   `position_embedding` field stays — it's used by the fixed-tile
   path and corresponds to the same 27×27 grid for the published
   checkpoint, so the loader can populate both from the same
   safetensors key (`embeddings.position_embedding.weight`).

2. **Bilinear position-embedding interpolator.** Port the
   `interpolate_pos_encoding(target_h, target_w)` logic from the
   eager `PatchEmbedding` (vision.rs:143-231). Keep the
   `align_corners=False` convention (matches PyTorch's
   `nn.functional.interpolate(mode='bilinear', align_corners=False)`).
   Two implementation choices:
   - **Option A: Host-side interpolation.** Compute the interpolated
     `(1, target_h*target_w, hidden_size)` tensor on the host
     (Vec<f32>), wrap as a `const_f32_like` LazyTensor. Matches what
     the eager code does (it was always host-side; the LFU cache
     reuses the result tensor handle). Cheap to implement; no new
     LazyTensor ops required.
   - **Option B: Graph-level interpolation.** Build the interpolated
     embedding as a chain of `slice` + `gather` + `mul_scalar` +
     `add` LazyTensor ops. More work; only useful if the
     interpolator's output ever needs gradient propagation (it
     doesn't — vision-encoder position embeddings are frozen during
     inference). **Pick Option A.**
   Add an LFU cache keyed on `(target_h, target_w)` mirroring the
   eager `PosEmbedCache` (vision.rs:20-76). Use `RefCell<HashMap>`
   to keep `&self.forward(...)` non-`mut`. Cap default = 16 entries
   (matches eager `DEFAULT_POS_EMBED_CACHE_SIZE`).

3. **2D RoPE table generation for arbitrary `(h_patches, w_patches)`.**
   The existing `build_2d_rope_tables` helper (in
   `lazy_paddleocr_vl_vision.rs`) computes tables for the fixed
   `patches_per_side × patches_per_side` grid. NaViT needs tables for
   `h_patches × w_patches`. Two options:
   - Generalize the existing helper to take `(h, w)` and emit the
     `num_patches × head_dim` table by raster-scanning rows + cols
     into the inv-freq pairs (eager `rot_pos_emb` at vision.rs:909-
     956 is the reference).
   - Add a new `build_2d_rope_tables_dynamic(theta, head_dim, h, w)`
     sibling. **Pick the generalization** — same name, optional `(h,
     w)` tuple replacing `patches_per_side`; the fixed-tile call site
     passes `(s, s)` and the NaViT call site passes the actual grid.

4. **Forward signature.**
   ```rust
   pub fn forward(
       &self,
       pixel_values: &LazyTensor, // (1, num_channels, H, W) where H, W % patch_size == 0
   ) -> Result<LazyTensor>;
   ```
   Return shape: `((H/patch_size * W/patch_size) / spatial_merge_size^2,
   text_hidden_size)`. No `tile_grid` parameter — the patch grid is
   derived from the input shape.

   Validate at build time: `H % patch_size == 0`, `W % patch_size
   == 0`, batch dim is 1 (NaViT batches by concatenating along the
   patch dim, not the leading dim — out of scope for v1; defer
   multi-image batching to a follow-up).

5. **Forward implementation.** Mirror the existing
   `PaddleOcrVlVisionModel::forward` body but skip the per-tile loop
   — there's a single conv-embed call across the whole `(1, C, H, W)`
   input. Add `interpolate_pos_encoding(h_patches, w_patches)` to
   produce the position embedding (vs the existing per-tile constant).
   The encoder loop, post-LN, and projector are identical — refactor
   them to a shared helper if the diff size warrants.

6. **`cu_seqlens` — defer.** The eager NaViT supports packing
   multiple images into a single sequence via variable-length
   attention (`cu_seqlens` cumulative-seqlen markers). The v1 NaViT
   processes one image per call. Variable-length packing is a
   follow-up that pairs with multi-image batching; flag in a
   `// TODO: cu_seqlens packing` comment for the audit trail.

7. **`chunked_attention` — port if needed.** The eager
   `chunked_attention` (vision.rs:299-387) ships the online-softmax
   tiled attention for very long sequences (14K+ patches). For
   document images at the default `max_pixels = 2_822_400`, the
   patch count peaks at ~14400 (= 2_822_400 / 14^2). That's right at
   the threshold where the chunked path matters for memory.
   **Decision:** if the existing `LazyTensor::softmax_last_dim` +
   `matmul` chain comfortably handles 14K KV positions on the
   test machine without OOM, skip the chunked path and document.
   Otherwise port the online-softmax tile loop to LazyTensor (it's
   ~50 lines of straightforward ops). **Engage critically** — if
   memory profiling shows it's needed, add it; don't ship without
   knowing.

8. **`forward_multi` / `forward_with_export` — out of scope.** The
   eager NaViT had multi-image and debug-export variants; the v1
   NaViT lazy port ships single-image forward only. Multi-image
   follows the `cu_seqlens` packing follow-up.

9. **Tests.** Mirror the existing
   `lazy_paddleocr_vl_vision.rs::tests::forward_shape_*` patterns:
   - `navit_forward_shape_at_base_grid` — `(1, 3, image_size,
     image_size)` input (= 27×27 patches at the published default),
     verify output shape equals
     `(num_patches/spatial_merge^2, text_hidden_size)` and all values
     finite.
   - `navit_forward_shape_landscape` — wider-than-tall input (e.g.
     `(1, 3, 14*8, 14*16)` = 8×16 patches), verify output shape
     scales correctly.
   - `navit_forward_shape_portrait` — taller-than-wide.
   - `navit_pos_embedding_interpolation_matches_eager` — port a
     small fixed input through the eager `interpolate_pos_encoding`
     (extracted as a standalone helper in the test) and compare
     output element-wise to the new LazyTensor path. Tolerance:
     1e-6 (pure host arithmetic on both sides).
   - `navit_pos_embedding_cache_hits` — call `forward` with the
     same `(H, W)` twice, assert the cache entry count went from
     0 to 1 after the first and stayed at 1 after the second.
   - `navit_panic_on_misaligned_input` — `(1, 3, 13, 14)`
     should fail at build time with a clear message about
     `H % patch_size != 0`.

10. **Loader.** Extend the existing `load_paddleocr_vl_vision_weights`
    function (or add a sibling `load_paddleocr_vl_navit_weights`) to
    populate `PaddleOcrVlNaVitWeights`. Use the same safetensors keys
    (`visual.vision_model.embeddings.position_embedding.weight`
    points at the same 27×27 grid; the `packing_position_embedding`
    fallback at the eager line 122-124 is unused by the lazy port —
    skip it).

**Files touched:**
- `fuel-core/src/lazy_paddleocr_vl_vision.rs` — add NaVit{Config,
  Weights, Model} alongside existing types; generalize
  `build_2d_rope_tables`; add `interpolate_pos_encoding` + LFU cache.
- `fuel-core/src/lazy_paddleocr_vl_vision.rs::tests` — add the
  test module entries above.

### Part 2 — Bilinear image preprocessor (~1 session)

**Goal:** add `fuel-core::lazy_paddleocr_vl::preprocess` submodule
with the canonical PaddleOCR-VL bilinear preprocessor. Surface as:

```rust
pub fn bilinear_resize_to_grid(
    image: &image::DynamicImage,
    supported_grids: &[(usize, usize)],
) -> (LazyTensor, usize, usize);
```

Returns `(pixels: LazyTensor of (3, H', W'), H', W')` where `H' × W'`
is the closest supported grid by aspect-ratio matching, and `pixels`
is ImageNet-normalized (mean = `[0.5, 0.5, 0.5]`, std = `[0.5, 0.5,
0.5]` for the PaddleOCR-VL preprocessor convention — verify against
the eager checkpoint's `preprocessor_config.json`; the historical
`/255 * 2 - 1` shorthand in the retired binary is exactly mean=0.5
std=0.5).

**Design:**

1. **Where the math lives.** Put the bilinear interpolation and
   normalization in `fuel-core::lazy_paddleocr_vl::preprocess` so any
   downstream binary or library can call it without re-deriving the
   formulas. The host-side bytes-to-f32 unpacking and `DynamicImage`
   decode stay in the binary (the binary is the only consumer that
   wants the `image` crate dependency).

2. **`supported_grids` semantics.** Each `(rows, cols)` is a multiple
   of `(patch_size * spatial_merge_size, patch_size * spatial_merge_size)`
   = `(28, 28)` for the published checkpoint. The selector picks the
   grid minimizing absolute aspect-ratio distance to the input's
   aspect ratio (the eager `smart_resize` algorithm at the deleted
   binary lines 209-260 is the reference). Specifically:
   - If input aspect ratio > 200, return an error
     (`Result<(LazyTensor, usize, usize)>`, not infallible).
     Matches eager (deleted binary line 234-240). The function
     signature changes to `Result<...>` if it's not already.
   - Clamp the target total pixel count to `[min_pixels,
     max_pixels]` (defaults `147_384`, `2_822_400`). Take these as
     additional function params with sensible defaults; consider a
     `BilinearResizeConfig` struct if the param list grows past 4.
   - Round `H` and `W` independently to the nearest multiple of
     `factor = patch_size * spatial_merge_size`.
   - If the rounded `(H, W)` falls outside `[min_pixels,
     max_pixels]`, rescale by the eager `beta` formula (deleted
     binary line 252-260).

3. **Bilinear resize.**
   - Use `image::imageops::resize(&img, new_w, new_h,
     FilterType::CatmullRom)` to do the host-side resize. The
     retired binary used `CatmullRom` (deleted binary line 290) —
     keep it for fidelity to the original. **Engage critically** —
     if the eager HuggingFace preprocessor used pure bilinear
     (`FilterType::Triangle`), prefer matching that. Check the
     reference `preprocessor_config.json` from
     `PaddlePaddle/PaddleOCR-VL` to settle.
   - Take an `&image::DynamicImage` as input; the function calls
     `.to_rgb8()` internally to normalize to 8-bit RGB.

4. **Normalization.**
   - Per-channel: `(pixel_byte / 255.0 - mean) / std`. Make `mean`
     and `std` params (default `[0.5; 3]` each); confirm against
     the upstream preprocessor config before locking the default.
   - Layout: CHW (channel-major). Matches the existing
     `forward_with_image` (`lazy_paddleocr_vl.rs`) input expectation.

5. **Output construction.** Wrap the normalized `Vec<f32>` in
   `LazyTensor::from_f32(Arc::from(data), Shape::from_dims(&[3, H',
   W']), &Device::cpu())`. Matches the existing input shape
   convention.

6. **Integration with `forward_with_image`.** Replace the
   `resize_nearest_chw` calls at `lazy_paddleocr_vl.rs:182-186`
   with a bilinear resize of each tile. **Decision point:** the
   existing fixed-tile path partitions the image first and then
   resizes each tile to `(image_size, image_size)`. With the NaViT
   path, the whole image is resized once to a grid that's
   patch-aligned, and the encoder consumes it directly — no tile
   partition. The cleaner shape is:
   - For `PaddleOcrVlModel::forward` with the NaViT encoder
     enabled, call `bilinear_resize_to_grid` once, pass result to
     `PaddleOcrVlNaVitModel::forward`. No tile partition, no per-
     tile resize.
   - The legacy `PaddleOcrVlVisionModel` (fixed tile grid) path
     stays available but is no longer the default. The bilinear
     preprocessor still wants to replace `resize_nearest_chw`
     inside that path's per-tile resize for OCR-quality reasons —
     do that as a small additional commit using
     `image::imageops::resize` on each tile's RGB byte buffer.
     Eager parity test from the retired binary's behavior is the
     bar.

7. **Add a `PaddleOcrVlModel::forward_navit(...)`** entry point
   that uses the new NaVit encoder + bilinear preprocessor. Keep
   the existing `forward(...)` working as-is for the fixed-tile
   path. Document in the doc comment that the NaVit path is the
   recommended one for OCR; the fixed-tile path stays for the
   simple cases and for downstream backwards compatibility.

8. **Tests.**
   - `bilinear_resize_to_grid_aspect_match` — for a 384×384 input
     and supported_grids = `[(28, 28), (56, 28), (28, 56)]`,
     verify the chosen grid is `(28, 28)`. Repeat for landscape +
     portrait + degenerate inputs.
   - `bilinear_resize_to_grid_pixel_clamp` — pass min_pixels +
     max_pixels that force the rescale branch; verify the output
     dims satisfy both the multiple-of-factor and the pixel-count
     constraints.
   - `bilinear_resize_aspect_ratio_error` — a 1×201 image should
     surface the aspect-ratio-too-large error.
   - `normalization_matches_eager` — feed a hand-constructed
     1×1×3 image of bytes `[128, 128, 128]`; expected output
     pixels with mean=0.5 std=0.5 are all `0.0` (= `(128/255 -
     0.5) / 0.5` ≈ `0.0039`). Hand-verify.
   - `forward_navit_smoke` — end-to-end smoke test on the existing
     tiny config: build a 28-pixel-multiple input, call
     `PaddleOcrVlModel::forward_navit`, assert the output logits
     shape is `(1, seq, vocab_size)` and values finite. Don't
     numerically compare to eager — the eager path is gone from
     the live build, so the smoke test is the contract.

**Files touched:**
- `fuel-core/src/lazy_paddleocr_vl.rs` — add the `preprocess`
  submodule (or break out to `lazy_paddleocr_vl_preprocess.rs`
  if file size is a concern; current file is ~858 LOC, adding
  ~200 keeps it under 1100 and matches sibling files).
- `fuel-core/src/lazy_paddleocr_vl.rs::tests` — preprocess
  unit tests + `forward_navit` smoke.
- `fuel-core/Cargo.toml` — verify `image` is a workspace dep or
  add as a fuel-core feature-gated dep. If image-loading vs.
  pure-math is split correctly, only the bilinear math needs to
  be in fuel-core; `DynamicImage` arrives pre-decoded from the
  caller, so `image::imageops::resize` is the only `image`-crate
  call inside fuel-core. Verify `imageops::resize` works on a
  raw `ImageBuffer<Rgb<u8>, Vec<u8>>` (it does), avoiding the
  full `DynamicImage` decode path inside fuel-core.

### Part 3 — Binary revival (~0.5 session)

**Goal:** rewrite `fuel-examples/examples/paddleocr-vl/main.rs` (renamed
from `_paddleocr-vl_retired/`) against the completed lazy modules.
The retired binary at git ref `cfcb35cf^` is the structural template;
the new binary differs in:

1. **Drops the `fuel_transformers::models::paddleocr_vl::Config`
   eager dep.** That HF config type was the explicit blocker quoted
   in commit `cfcb35cf`'s release notes ("paddleocr-vl — still uses
   eager Config helper"). Replace with either:
   - A purpose-built `HfConfig` struct in the binary that
     deserializes the safetensors-side config.json, OR
   - A `fuel-core::lazy_paddleocr_vl::hf_config` module that lives
     alongside the lazy model definition. **Pick the second** — the
     HF config translation is canonical to the model, not the binary.
     Add `HfPaddleOcrVlConfig` + `From<&HfPaddleOcrVlConfig> for
     PaddleOcrVlConfig` to `fuel-core/src/lazy_paddleocr_vl.rs`.
     This pattern is what other migrated lazy ports use (cross-
     check `fuel-core/src/lazy_llama.rs` or similar for prior art).
2. **Uses the NaVit forward path by default.** `forward_navit`
   instead of `forward`. The fixed-tile path can be an opt-in
   flag (`--legacy-tile-grid`) for debugging.
3. **Uses the bilinear preprocessor** for both the NaVit path and
   the legacy fixed-tile path's per-tile resize.
4. **Restores the rename of the directory.** `mv
   fuel-examples/examples/_paddleocr-vl_retired
   fuel-examples/examples/paddleocr-vl` is the activation step;
   Cargo auto-discovery picks up `main.rs` from there once present.

**Scope of the v1 binary:**

- `--image` single-image mode only, matching the retired binary's
  v1 state. Multi-image, batch, and video remain CLI flags but
  return an error like the retired version did.
- `--task ocr/table/formula/chart` is just prompt selection (same
  as retired).
- `--cpu` and `--bf16` flags accepted but no-op (lazy v1 is CPU-F32,
  same as retired). Document in the README.
- Single greedy-argmax next-token step is the contract. `--max-length`
  is accepted for CLI parity but only the first generated token is
  printed — same as retired. A real generation loop (KV cache + EOS)
  is a separate session under
  `docs/session-prompts/remaining-eager-ports-tracker.md` follow-up.

**Tests:**

- No unit tests on the binary itself.
- Manual smoke test: run against each of the
  `fuel-examples/examples/paddleocr-vl/test_*.png` fixtures (they
  survived the retirement). Verify the binary runs end-to-end,
  loads weights, encodes the image, and produces a finite logits
  tensor. Cross-check against published HuggingFace inference for
  task=ocr on `test_ocr.png` — same first token (or top-3) is the
  realistic bar without a generation loop.

**Files touched:**
- `fuel-examples/examples/_paddleocr-vl_retired/` → rename to
  `fuel-examples/examples/paddleocr-vl/`.
- `fuel-examples/examples/paddleocr-vl/main.rs` — new, ~250-300 LOC
  (smaller than the retired 457 LOC since `smart_resize` + image
  loader move out of the binary into fuel-core).
- `fuel-examples/examples/paddleocr-vl/README.md` — update to
  reflect that NaVit is the default, fixed-tile is a legacy flag,
  multi-image/batch/video remain unimplemented and surface a clear
  error.
- `fuel-core/src/lazy_paddleocr_vl.rs` — add `HfPaddleOcrVlConfig`
  + `From` impl.

## What's NOT in scope

- **Multi-image / batch / video modes.** All three remain
  unimplemented and return a clear `not implemented` error from
  the CLI. Each is its own session (multi-image wants `cu_seqlens`
  packing in NaVit; video wants temporal RoPE in the text path,
  which the lazy text port may or may not have).
- **Generation loop with KV cache and EOS detection.** The lazy
  text port (`lazy_paddleocr_vl_text`) may already have the
  per-step forward shape needed — if so, queue the gen loop as
  immediate follow-up. If not, KV cache integration is its own
  session under `docs/session-prompts/eager-tensor-retirement-master-plan.md`.
- **F16 / BF16 dtype expansion.** Lazy path is F32. BF16/F16
  expansion for the vision encoder is plausible (F16 unary +
  binary CPU + the existing CUDA/Vulkan kernels exist) but the
  text path may have gaps; defer to its own session.
- **CUDA / Vulkan dispatch for the NaVit encoder ops.** The new
  forward path uses only ops that already have non-CPU dispatch
  (conv2d, layer_norm, matmul, slice, concat, permute), so the
  binary will work on CUDA/Vulkan if `--cpu` is dropped. Live
  hardware test on RTX 4070 is a nice-to-have during the binary
  session but not required.
- **Replacing the LFU cache with the Phase 7.6 op cache.** The
  position-embedding interpolation cache is intentionally a
  domain-specific `RefCell<HashMap>` mirroring the eager code.
  General-purpose LazyTensor result caching is a separate
  architectural question (cf. `project_phase_7_6_step_*.md`).
- **`packing_position_embedding` fallback.** The eager code carries
  a 32768-position fallback table (vision.rs:122-124) that's
  unused for the published checkpoint. Skip it.

## Scope estimate

- **Part 1 (NaVit port):** 4-6 hours. The bulk is mechanical mirror
  of the eager `PatchEmbedding` interpolation logic + the
  `build_2d_rope_tables` generalization; the encoder loop is
  unchanged. Tests are ~1 hour.
- **Part 2 (bilinear preprocessor):** 2-3 hours. Smart-resize
  math is small; the win is having it in fuel-core where the
  next vision-lang model that wants it can pick it up.
- **Part 3 (binary revival):** 2-3 hours. Most of the time is in
  the `HfPaddleOcrVlConfig` deserialization + the prompt-template
  assembly + the smoke-test verification.
- **Total: 2-3 sessions** depending on packing. A tight session can
  ship Part 1 + Part 2 together; Part 3 fits in a half-session
  follow-up. A relaxed pace splits Part 1 and Part 2 into separate
  sessions and folds Part 3 into Part 2.

## Coordination

- **`fuel-transformers/_models_retired/`.** The eager NaViT
  reference lives in the retired tree. If the
  `eager-retirement-phase-h-plan.md` cleanup sweep deletes that
  tree (vs. leaving it git-history-only), do this port first.
  Check `git log --since="14 days" fuel-transformers/src/_models_retired/`
  for activity.
- **`docs/session-prompts/remaining-eager-ports-tracker.md`.**
  PaddleOCR-VL is listed there as a quarantined binary. Update
  that tracker after this session ships to remove it from the
  pending list.
- **`docs/session-prompts/eager-tensor-retirement-master-plan.md`.**
  The master plan tracks the LLM family and the multimodal family
  separately. The completion of PaddleOCR-VL moves the multimodal
  count by one — update the running count.
- **Image dependency.** If fuel-core gains a non-feature-gated
  `image` dep, surface that decision — the conservative call is
  to gate behind a `paddleocr-vl-preprocess` feature so library
  consumers who only want the text path don't pull in `image` +
  `png` + `jpeg`. Engage critically.

## References

- Memory: `project_multimodal_compositions_shipped.md` — establishes
  the vision-language composition pattern (PaliGemma + LLaVA), which
  PaddleOCR-VL extends; the `forward_embeds(...)` recipe used by the
  text model is shared.
- Memory: `project_vision_multimodal_foundations_shipped.md` — the
  ViT / CLIP / SigLIP / DINOv2 ports that built the lazy vision
  encoder vocabulary. NaViT is the next variant on the stack.
- Memory: `project_phase_d_batch_b_shipped.md`,
  `project_phase_d_batch_e_shipped.md` — pattern library for the
  binary's HF-config-translation + safetensors-loading shape.
- Commits to study:
  - `cfcb35cf` — the retirement commit; recovers the deleted
    `main.rs` via `git show cfcb35cf -- fuel-examples/examples/paddleocr-vl/main.rs`.
  - The original PaddleOCR-VL landing: `255ade6f` (lazy module
    landings) and the prior eager binary in `f526033d`.
- Session prompts:
  - `docs/session-prompts/eager-retirement-phase-h-plan.md` — the
    plan that documented this binary's deferral.
  - `docs/session-prompts/remaining-eager-ports-tracker.md` —
    where this work item is queued.
  - `docs/session-prompts/eager-tensor-retirement-master-plan.md`
    — the umbrella program.
- Upstream:
  - HuggingFace model card:
    `https://huggingface.co/PaddlePaddle/PaddleOCR-VL`.
  - Paper: `arXiv:2510.14528`.
  - Reference Python `preprocessor_config.json` — read for the
    canonical `image_mean` / `image_std` values before locking
    the defaults in the lazy preprocessor.
