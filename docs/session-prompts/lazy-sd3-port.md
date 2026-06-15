# Port: Stable Diffusion 3 / 3.5 (lazy_sd3 family) — revive `_stable-diffusion-3_retired`

> **Status (reconciled 2026-06-15 against the 2026-06-14 redirection + current git).** Round 1 **shipped** in commit `7c50b221` (`feat(lazy): Phase 7 — finish remaining retirement`): `Sd3TripleClip` (`lazy_sd3_text_encoder`), `SdVae3Decoder` (`lazy_sd3_vae`), `flow_match_euler_sample` + SLG (`lazy_sd_samplers_sd3`), and `MmDitFullModel` with a `skip_layers` parameter (`lazy_mmdit`). This covers SD3-medium and SD 3.5-large / large-turbo (standard joint blocks).
>
> **§1.3.3 (MMDiT-X joint block for SD 3.5-medium with skip-layer guidance) is still TODO** — explicit code markers live at `fuel-core/src/lazy_mmdit.rs:746`, `:825-826`, and `:1437-1445`. A repo grep shows **this prompt is the only record of that plan**, so this doc remains the live design home for the MMDiT-X follow-up; do not retire it.
>
> Note: `MmDitFullConfig::sd3_5_medium()` + the binary's `--use-slg` path is **wired-but-incorrect** — it currently runs the non-X joint-block path (see the `:825-826` comment) and will not produce correct SD 3.5-medium output until MMDiT-X lands.

## Why this is needed

Phase H of the eager-retirement program ([commit `cfcb35cf`](../../fuel-examples/examples/_stable-diffusion-3_retired/README.md)) moved every `fuel-transformers/src/models/*` source tree out of the workspace build and into `fuel-transformers/src/_models_retired/`. Binaries whose lazy port was structurally incomplete were quarantined by renaming their example directory `_<name>_retired/` so Cargo auto-discovery skips them.

`fuel-examples/examples/_stable-diffusion-3_retired/` is one of those quarantined binaries. The Phase H commit message lists it as:

> `stable-diffusion-3` — needs lazy_sd3 family (triple-CLIP + VAE + sampler)

The old eager binary lived at `fuel-examples/examples/stable-diffusion-3/` and consisted of four files (`main.rs` 273 LOC + `clip.rs` 234 LOC + `vae.rs` 93 LOC + `sampling.rs` 84 LOC); it imported:

- `fuel_transformers::models::mmdit::model::{Config, MMDiT}` — the SD3 MMDiT-X transformer (now retired under `fuel-transformers/src/_models_retired/diffusion/mmdit/*`).
- `fuel_transformers::models::stable_diffusion::clip::{ClipTextTransformer, Config}` — the CLIP-L + CLIP-G text encoders (now retired under `fuel-transformers/src/_models_retired/diffusion/stable_diffusion/clip.rs`).
- `fuel_transformers::models::stable_diffusion::vae::{AutoEncoderKL, AutoEncoderKLConfig}` — the SD VAE with 16 latent channels (now retired under `fuel-transformers/src/_models_retired/diffusion/stable_diffusion/vae.rs`).
- `fuel_transformers::models::t5::T5EncoderModel` — the T5-XXL text encoder (now retired).
- `fuel_transformers::models::flux::sampling::get_noise` — the noise generator (reused by Flux; the lazy_flux port already covers this).

The eager source is the reference. We will not revive it; we port to lazy modules and rewrite the binary against the lazy API.

## What already exists in the lazy world

These are the pieces we **do not** need to re-port:

- `fuel-core/src/lazy_mmdit.rs` (980 LOC) — SD3-style DoubleStreamBlock + SingleStreamBlock + AdaLN modulation; `MmDitModel::forward(img, txt, timestep, y)` accepts patchified image tokens shaped `(B, S_image, dim)` and conditioning vectors; `load_from_mmapped` is wired. **Gaps for SD3** are documented in §3.3 below.
- `fuel-core/src/lazy_clip.rs` (1020 LOC) — CLIP text-encoder substrate. Provides the per-layer attention + MLP machinery used by CLIP-L (sdxl_te1) and CLIP-G (sdxl_te2). Building on this is correct.
- `fuel-core/src/lazy_t5.rs` (1101 LOC) — T5 encoder. Should provide the T5-XXL encoder T5-v1_1 forward used here.
- `fuel-core/src/lazy_sd_vae.rs` (687 LOC) — SD 1.5 VAE decoder. Config-driven (`SdVaeConfig::latent_channels`), 4-channel-default. **Gaps for SD3** are documented in §3.2 below.
- `fuel-core/src/lazy_sd_text_encoder.rs` (846 LOC) — single CLIP text encoder for SD 1.5 / SDXL TE1 / SDXL TE2. The triple-CLIP composer that SD3 needs is **not** here.
- `fuel-core/src/lazy_sd_samplers_euler.rs` (512 LOC) — Euler-ancestral discrete scheduler (SD-1.5 / SDXL noise schedule). **Not** the SD3 flow-match Euler with time_snr_shift + SLG; that is a different sampler.
- `fuel-core/src/lazy_flux.rs` (1958 LOC) — Flux model + autoencoder + flow-matching sampler. The flow-match Euler loop is close in spirit but tied to Flux's signature (no SLG, different MMDiT). Worth reading for shape inspiration but **not** the substrate to extend.

## 1. Required new lazy modules (target surface)

### 1.1 `fuel_core::lazy_sd3_text_encoder` — triple-CLIP composer

New file: `fuel-core/src/lazy_sd3_text_encoder.rs`.

SD3 text conditioning is the concatenation of three text-encoder outputs:

- **CLIP-L** (`openai/clip-vit-large-patch14`, 768-dim, 12 layers) — penultimate-layer hidden states (`(B, 77, 768)`) + EOS-pooled embedding (`(B, 768)`).
- **CLIP-G** (`laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`, 1280-dim, 32 layers) — penultimate-layer hidden states (`(B, 77, 1280)`) + EOS-pooled embedding (`(B, 1280)`) projected through a `[1280, 1280]` linear (no bias).
- **T5-XXL** (`google/t5-v1_1-xxl`, 4096-dim, 24 layers) — full encoder output (`(B, 77, 4096)`).

The composer assembles two tensors:

- `y` — pooled vector `(B, 2048)` = `cat([clip_l_pooled, clip_g_pooled_projected], dim=-1)`.
- `context` — per-token sequence `(B, 154, 4096)` =
  1. `clip_concat = cat([clip_l_hidden, clip_g_hidden], dim=-1)` → `(B, 77, 2048)`,
  2. `clip_padded = pad_with_zeros(clip_concat, dim=-1, 0, 2048)` → `(B, 77, 4096)`,
  3. `cat([clip_padded, t5_hidden], dim=-2)` → `(B, 154, 4096)`.

Eager reference: `fuel-examples/examples/_stable-diffusion-3_retired/clip.rs` (recovered from git history at `cfcb35cf~1:fuel-examples/examples/stable-diffusion-3/clip.rs`), 234 LOC. Defines `ClipWithTokenizer`, `T5WithTokenizer`, and `StableDiffusion3TripleClipWithTokenizer::{new, new_split, encode_text_to_embedding}`.

#### Suggested API

```rust
pub struct Sd3TripleClip {
    clip_l: lazy_clip::ClipTextModel,
    clip_g: lazy_clip::ClipTextModel,
    clip_g_text_projection: WeightStorage, // [1280, 1280] no-bias linear
    t5: lazy_t5::T5EncoderModel,
}

impl Sd3TripleClip {
    pub fn load_from_mmapped_monolithic(
        st: &SafeTensors,
        // for SD3-medium: weights live under `text_encoders.clip_l.transformer`,
        // `text_encoders.clip_g.transformer`, `text_encoders.t5xxl.transformer`.
    ) -> Result<Self>;

    pub fn load_from_mmapped_split(
        st_clip_l: &SafeTensors,
        st_clip_g: &SafeTensors,
        st_t5: &SafeTensors,
        // for SD 3.5: separate `clip_l.safetensors` / `clip_g.safetensors` /
        // `t5xxl_fp16.safetensors` files; CLIP-G's `text_projection` lives in
        // its own file at the root, not under `transformer`.
    ) -> Result<Self>;

    /// Returns `(context, y)`:
    ///   - `context: (B, 154, 4096)` — per-token conditioning fed into MMDiT
    ///     context_embedder.
    ///   - `y: (B, 2048)` — pooled conditioning fed into MMDiT y_embedder.
    pub fn encode(
        &self,
        clip_l_tokens: &[u32],   // 77 ids, pre-padded with EOS or pad-id
        clip_g_tokens: &[u32],
        t5_tokens: &[u32],
    ) -> Result<(LazyTensor, LazyTensor)>;
}
```

Tokenizer ownership: keep tokenizers in the **binary**, not in the model. The lazy_* modules should stay pure-tensor; the binary tokenizes once and hands `[u32]` slices to `encode`. This matches the convention already used by `lazy_clip` / `lazy_t5`.

Open questions to resolve in-session:

- CLIP-L expects QuickGELU (the `quick_gelu` activation in `lazy_sd_text_encoder::ClipActivation`); CLIP-G's eager config (`Config::sdxl2`) likewise uses QuickGELU. Both should reuse `lazy_clip` rather than `lazy_sd_text_encoder` for the new triple wrapper — verify which substrate the SD3 forward (`forward_until_encoder_layer(..., -2)`, i.e. penultimate-layer extraction) maps onto more cleanly before committing.
- Confirm the F16 vs F32 dtype boundary. Eager loads safetensors as `F16` for all three encoders but casts the T5 output to `F32` via `t5.forward_dt(_, Some(DType::F32))?` before casting back to `F16` at concat time. The lazy port should reuse `lazy_t5`'s dtype-parameterized forward.

### 1.2 `fuel_core::lazy_sd3_vae` — 16-channel VAE decoder

New file: `fuel-core/src/lazy_sd3_vae.rs`, **or** SD3-specific variant inside `lazy_sd_vae` (preferred: new file because the eager source already separated the SD3 config and the up_block layout differs).

SD3 uses the same AutoencoderKL family as SD 1.5 but with three knobs flipped:

- `latent_channels: 16` (vs 4 in SD 1.5).
- `block_out_channels: [128, 256, 512, 512]` (vs `[128, 256, 512, 512]` for SD 1.5 — same here, but be aware that SD3 also has `use_quant_conv: false` and `use_post_quant_conv: false`).
- The eager weight layout uses a `vb_rename` callback to map HF safetensors names to the legacy SD layout: `down_blocks → down`, `up_blocks → up.{3,2,1,0}` (reversed), `resnets → block`, `attentions.0 → attn_1`, `query/key/value → q/k/v`, `proj_attn → proj_out`, `conv_norm_out → norm_out`, `downsamplers → downsample`, `upsamplers → upsample`, `conv_shortcut → nin_shortcut`. See the eager `sd3_vae_vb_rename` fn (recovered from git: `cfcb35cf~1:fuel-examples/examples/stable-diffusion-3/vae.rs`, lines 17-93).

Eager reference for the VAE itself: `fuel-transformers/src/_models_retired/diffusion/stable_diffusion/vae.rs` (409 LOC) — defines `AutoEncoderKLConfig` + `Decoder` + `Encoder` + `AutoEncoderKL`.

#### Suggested API

```rust
pub struct SdVae3Decoder {
    weights: SdVaeDecoderWeights, // can share with SdVaeDecoder if shape matches
    config: SdVae3Config,
}

#[derive(Debug, Clone)]
pub struct SdVae3Config {
    /// Decoder-order channel widths. SD3 = `[512, 512, 512, 256, 128]`
    /// (4 up-blocks stepping down from 512 to 128, identical shape to SD 1.5).
    pub dims: Vec<usize>,
    pub latent_channels: usize,     // 16 for SD3
    pub layers_per_block: usize,    // 3 for SD3 (= layers_per_block 2 + 1 final, per eager)
    pub norm_num_groups: usize,     // 32
    pub use_post_quant_conv: bool,  // false for SD3
}

impl SdVae3Decoder {
    pub fn load_from_mmapped(st: &SafeTensors, cfg: &SdVae3Config) -> Result<Self>;
    pub fn decode(&self, latents: &LazyTensor) -> Result<LazyTensor>;
}
```

Reuse pattern: most building blocks (`group_norm`, `conv2d_k3_s1_p1`, `conv2d_k1_s1_p0`, `upsample_nearest_2x`, `vae_spatial_attention`) already exist in `lazy_sd_vae`. The new module is primarily a different config + a different `conv_in` shape (`[1, 16, H, W] → [1, 512, H, W]` instead of `[1, 4, ...] → [1, 512, ...]`) + a different number of decoder layers per up-block. Decide in-session whether to:

(a) **Extend `lazy_sd_vae`** by parameterizing `SdVaeConfig::layers_per_block` (already a field, but always 2; SD3 wants `layers_per_block: 2` per the eager config — verify), `use_post_quant_conv`, and document the HF-name renaming. Then SD3 is just a different config + a name-mapping helper.

(b) **New file `lazy_sd3_vae.rs`** that re-uses the building-block fns from `lazy_sd_vae` (promote them to `pub` if needed) but owns its own weight struct + load fn.

Recommendation: **(a)** unless the per-up-block layer count actually differs. The eager `Decoder` has `num_layers: config.layers_per_block + 1` regardless of SD1/SD3 — that's the same `+1` rule, so `lazy_sd_vae` likely already gets the layer count right. Read both during the port to confirm.

### 1.3 `fuel_core::lazy_mmdit` — extend the existing module

The shipped `MmDitModel::forward(img, txt, timestep, y)` accepts **already-patchified** image tokens, **already-embedded** text tokens, and applies one DoubleStream + one SingleStream stack with depth = `cfg.depth_double` + `cfg.depth_single`. Three SD3-specific gaps remain:

#### 1.3.1 Patchify / unpatchify wrapper + 2D positional embedding

The eager `MMDiT::forward(x: (N,C,H,W), t, y, context, skip_layers)` does, in order:

1. `cropped_pos_embed = pos_embedder.get_cropped_pos_embed(h, w)` — slices a `(pos_embed_max_size, pos_embed_max_size, hidden)` table at the right offset for the image's (h_patches, w_patches) extent.
2. `x = patch_embedder(x) + cropped_pos_embed` — `PatchEmbedder` is a `Conv2d(in_ch, hidden, kernel=patch_size, stride=patch_size, padding=0)`; output reshaped to `(N, H*W/patch², hidden)`.
3. Builds `c = timestep_embedder(t) + vector_embedder(y)`.
4. `context = context_embedder(context)` — `Linear(context_embed_size, hidden)`.
5. Runs `MMDiTCore::forward(context, x, c, skip_layers)`.
6. `unpatchifier.unpatchify(x, h, w)` — inverse of patch embed: `(N, num_patches, patch²*out_ch) → (N, out_ch, h_patches*patch, w_patches*patch)`.
7. Crops `narrow(2, 0, h).narrow(3, 0, w)`.

Eager reference: `fuel-transformers/src/_models_retired/diffusion/mmdit/{model.rs (256 LOC), embedding.rs (209 LOC), blocks.rs (522 LOC), projections.rs (131 LOC)}`.

The lazy MmDiT should grow either:

- A new `MmDitFullModel` wrapper that owns `PatchEmbedder` + `PositionEmbedder` + `TimestepEmbedder` + `VectorEmbedder` + `ContextEmbedder` + `Unpatchifier` + an inner `MmDitModel`, **or**
- An expanded `MmDitModel::forward_image` (image-space variant) alongside the current `forward` (token-space) so the existing tests continue to pass.

Recommendation: separate `pub struct MmDitFullModel` to keep the two surfaces independently testable. Mirrors how `lazy_flux` is laid out (model + autoencoder + sampling each their own type).

#### 1.3.2 `skip_layers: Option<&[usize]>` parameter

The SLG sampler (§1.4) requires `forward` to optionally skip a list of DoubleStream block indices. Eager wires this through `MMDiTCore::forward` (model.rs:243):

```rust
for (i, joint_block) in self.joint_blocks.iter().enumerate() {
    if let Some(skip_layers) = &skip_layers {
        if skip_layers.contains(&i) {
            continue;
        }
    }
    (context, x) = joint_block.forward(&context, &x, c)?;
}
```

Add the same loop guard to `lazy_mmdit::MmDitModel::forward`. Default `None` keeps the existing test signature; SLG callers pass `Some(&[7, 8, 9])` for SD 3.5-medium.

#### 1.3.3 MMDiT-X joint-block variant (SD 3.5-medium only)

SD 3.5-medium uses MMDiT-X blocks (one extra self-attention inside the image stream). The eager `model.rs:200` detects them at load time via `vb.contains_tensor("joint_blocks.{i}.x_block.attn2.qkv.weight")`. The lazy port can mirror this with a `MmDitBlockKind::Standard | MmDitX` enum per block, set during `load_from_mmapped`. Defer this to a follow-up sub-port if the first integration only targets SD3-medium and SD 3.5-large (which use the standard variant); document the gap if so.

#### 1.3.4 SD3 config presets

Add to `MmDitConfig`:

```rust
impl MmDitConfig {
    pub fn sd3_medium() -> Self { /* patch_size: 2, in: 16, depth: 24, head: 64, adm_in: 2048, pos_max: 192, ctx_embed: 4096, freq: 256 */ }
    pub fn sd3_5_medium() -> Self { /* ...pos_max: 384, MMDiT-X blocks */ }
    pub fn sd3_5_large() -> Self  { /* depth: 38 */ }
}
```

Eager presets live at `fuel-transformers/src/_models_retired/diffusion/mmdit/model.rs:31-76`.

### 1.4 `fuel_core::lazy_sd_samplers_sd3` — flow-match Euler + SLG

New file: `fuel-core/src/lazy_sd_samplers_sd3.rs` (preferred — distinct sampler family, mirrors `lazy_sd_samplers_euler` / `lazy_sd_samplers_unipc` split convention).

The SD3 sampler is flow-matching Euler with two SD3-specific knobs:

- **time_snr_shift** — non-linear remap of the linear t-schedule:

  ```rust
  fn time_snr_shift(alpha: f64, t: f64) -> f64 {
      alpha * t / (1.0 + (alpha - 1.0) * t)
  }
  ```

  Default `alpha = 3.0`. The SD3 paper's "Resolution-dependent shifting of timestep schedules" (https://arxiv.org/pdf/2403.03206).

- **CFG (classifier-free guidance)** — standard `cfg_scale * cond - (cfg_scale - 1) * uncond`.

- **Skip Layer Guidance (SLG)** — for SD 3.5-medium only. Within a configurable timestep window, runs the model a third time with a subset of DoubleStream blocks skipped, and pushes guidance toward the non-skipped output:

  ```rust
  pub struct SkipLayerGuidanceConfig {
      pub scale: f64,    // default 2.5
      pub start: f64,    // default 0.01 (fraction of total steps)
      pub end: f64,      // default 0.2
      pub layers: Vec<usize>, // default [7, 8, 9] for SD 3.5-medium
  }
  ```

Eager reference: `fuel-examples/examples/_stable-diffusion-3_retired/sampling.rs` (recovered from git: `cfcb35cf~1:...`, 84 LOC) — defines `SkipLayerGuidanceConfig` + `euler_sample` + `time_snr_shift` + `apply_cfg`.

#### Suggested API

```rust
pub struct SkipLayerGuidanceConfig { /* as above */ }

/// SD3 flow-match Euler sampler with optional SLG.
///
/// - `mmdit`: model with `forward(img: (N,C,H,W), t: (N,), y, context, skip_layers)`.
///   Caller hands the SD3 `MmDitFullModel` from §1.3.
/// - `y`: pooled conditioning `(2, 2048)` — `[cond, uncond]` stacked along batch.
/// - `context`: per-token conditioning `(2, 154, 4096)` — same stack.
/// - Returns: final latent `(1, 16, H/8, W/8)` ready for VAE decode.
#[allow(clippy::too_many_arguments)]
pub fn flow_match_euler_sample(
    mmdit: &MmDitFullModel,
    y: &LazyTensor,
    context: &LazyTensor,
    num_inference_steps: usize,
    cfg_scale: f64,
    time_shift: f64,        // alpha for time_snr_shift; 3.0 default
    height: usize,
    width: usize,
    slg_config: Option<SkipLayerGuidanceConfig>,
    seed: Option<u64>,
) -> Result<LazyTensor>;
```

Implementation skeleton (translating eager `sampling.rs:18-67` line-for-line):

1. `x = get_noise(1, height, width, device)` cast to F16 — initial latent `(1, 16, h/8, w/8)`. Note `get_noise` exists in `lazy_flux::sampling::get_noise`; reuse or hoist into a shared helper module.
2. Build `sigmas = (0..=N).map(|i| time_snr_shift(alpha, (N - i) as f64 / N as f64))`.
3. For each `(s_curr, s_prev)` window of `sigmas`:
   - `timestep = s_curr * 1000.0`.
   - `noise_pred = mmdit.forward(cat([x, x], 0), full(timestep, (2,)), y, context, None)` — two-way batch for CFG.
   - `guidance = cfg_scale * noise_pred[0:1] - (cfg_scale - 1) * noise_pred[1:2]`.
   - If `slg_config` present and `N * start < step < N * end`:
     - `slg_noise_pred = mmdit.forward(x, full(timestep, (1,)), y[0:1], context[0:1], Some(slg_config.layers))`.
     - `guidance += slg_config.scale * (noise_pred[0:1] - slg_noise_pred[0:1])`.
   - `x = x + guidance * (s_prev - s_curr)`.
4. Return `x`.

Primitives needed: all exist (concat, narrow, full, broadcast mul/sub/add). No new graph ops.

## 2. Per-module work items with file paths

| Sub-port | New / extend | Path | Eager reference (`_models_retired/`-rooted unless noted) | Approx LOC |
|----------|--------------|------|----------------------------------------------------------|------------|
| 2.1 | new file | `fuel-core/src/lazy_sd3_text_encoder.rs` | `fuel-examples/examples/_stable-diffusion-3_retired/clip.rs` (history; 234 LOC) + `diffusion/stable_diffusion/clip.rs` (441 LOC) | ~400 |
| 2.2 | extend / new | `fuel-core/src/lazy_sd_vae.rs` (extend) **or** `fuel-core/src/lazy_sd3_vae.rs` (new) | `diffusion/stable_diffusion/vae.rs` (409 LOC) + `_stable-diffusion-3_retired/vae.rs` (history; 93 LOC) | ~150 if extend, ~300 if new file |
| 2.3 | extend | `fuel-core/src/lazy_mmdit.rs` — add `MmDitFullModel`, `skip_layers`, `sd3_*` config presets, optional MMDiT-X support | `diffusion/mmdit/{model.rs (256), embedding.rs (209), blocks.rs (522), projections.rs (131)}` | ~400-600 |
| 2.4 | new file | `fuel-core/src/lazy_sd_samplers_sd3.rs` | `_stable-diffusion-3_retired/sampling.rs` (history; 84 LOC) | ~150 |
| 2.5 | rewrite | `fuel-examples/examples/_stable-diffusion-3_retired/main.rs` → `fuel-examples/examples/stable-diffusion-3/main.rs` (un-quarantine) | `_stable-diffusion-3_retired/main.rs` (history; 273 LOC) | ~250 |

After 2.5, rename the example directory back: `_stable-diffusion-3_retired/` → `stable-diffusion-3/`.

## 3. Binary revival (`fuel-examples/examples/stable-diffusion-3/main.rs`)

Rewrite against the lazy API. Concrete steps:

1. Replace `use fuel_transformers::models::mmdit::model::{Config as MMDiTConfig, MMDiT}` with `use fuel_core::lazy_mmdit::{MmDitConfig, MmDitFullModel}`.
2. Replace `crate::clip::StableDiffusion3TripleClipWithTokenizer` with `fuel_core::lazy_sd3_text_encoder::Sd3TripleClip` + a tokenizer trio constructed in-binary. Tokenization stays in the binary (matches the existing pattern in `fuel-examples/examples/stable-diffusion/main.rs` and `flux/main.rs`).
3. Replace `crate::vae::build_sd3_vae_autoencoder` + `sd3_vae_vb_rename` with `fuel_core::lazy_sd_vae::SdVae3Decoder::load_from_mmapped` (or `lazy_sd3_vae::...` if §1.2 chose path (b)).
4. Replace `crate::sampling::euler_sample` with `fuel_core::lazy_sd_samplers_sd3::flow_match_euler_sample`. Replace `crate::sampling::SkipLayerGuidanceConfig` with `fuel_core::lazy_sd_samplers_sd3::SkipLayerGuidanceConfig`.
5. Replace `fuel_nn::VarBuilder::from_mmaped_safetensors` with the safetensors-loader helpers already shipped per the round-7b lazy migrations (`fuel-core/src/lazy.rs` + module-specific `load_from_mmapped` fns). Verify the exact loader pattern by reading a recently-migrated example such as `fuel-examples/examples/flux/main.rs` (Phase D batch B).
6. Replace `fuel::Tensor` with `fuel_graph::Tensor` / `LazyTensor` for the concat / I/O dance:
   - `let context = Tensor::cat(&[context, context_uncond], 0)?` → `let context = context.concat(&context_uncond, 0_usize)?`.
   - `let y = Tensor::cat(&[y, y_uncond], 0)?` → analogous.
   - `autoencoder.decode(&((x / 1.5305)? + 0.0609)?)` → `autoencoder.decode(&x.div_scalar(1.5305)?.add_scalar(0.0609)?)` (or whichever scalar-arith helpers exist).
7. Replace `((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(fuel::DType::U8)?` with the lazy equivalent + a realize call before `fuel_examples::save_image`. (Image save needs CPU bytes; realize at the boundary.)
8. Keep the CLI args (`Which`, `--use-slg`, etc.) and HF download logic. Those are pure host code.

Once the binary builds + runs against a tiny config (see §5 test strategy), rename `_stable-diffusion-3_retired/` → `stable-diffusion-3/` so Cargo picks it up.

## 4. Suggested sequencing

Five sub-ports, ordered by dependency + risk:

1. **`lazy_sd3_text_encoder`** (1 session) — smallest, fewest unknowns; reuses `lazy_clip` + `lazy_t5` substrates that already exist and are mature. Lands first so downstream sub-ports can test text conditioning end-to-end.
2. **`lazy_sd_vae` extension (or `lazy_sd3_vae` new)** (1 session) — moderate; mostly config plumbing + a new `conv_in` shape. The HF→legacy name rename is a 50-line helper. Lands second because §2.3 doesn't depend on it.
3. **`lazy_mmdit` extension** (1-2 sessions) — biggest single sub-port; touches existing tested code. Split if MMDiT-X support is in scope; defer MMDiT-X to a follow-up if only SD3-medium + SD 3.5-large are targeted in the first round. Lands third.
4. **`lazy_sd_samplers_sd3`** (1 session) — small, pure host control + existing tensor ops. No new graph primitives. Lands fourth.
5. **Binary revival + un-quarantine** (1 session) — wire the four lazy modules together; debug the end-to-end image-generation smoke test against the published SD3-medium checkpoint at a tiny resolution (e.g. 256×256, 4 inference steps).

**Total: 5-6 sessions.** Add 1 if MMDiT-X support is in-scope for round 1; add 1 if backend dispatch (CUDA / Vulkan) for SD3-specific ops needs separate work.

## 5. Test strategy (per sub-port)

- **`lazy_sd3_text_encoder`**: shape test on `encode(&[u32]*77)` → `(context: (1, 154, 4096), y: (1, 2048))`. Golden test: hand-tokenize a fixed prompt with `tokenizers` crate, encode, assert first-row pooled vec matches an eager-extracted reference (or the published SD3 reference impl's output).
- **`lazy_sd3_vae`**: round-trip shape test: encode-and-decode a `(1, 3, 256, 256)` zero tensor (decoder-only is fine — just verify the decode path on a `(1, 16, 32, 32)` zero latent produces `(1, 3, 256, 256)`).
- **`lazy_mmdit::MmDitFullModel`**: tiny config (`patch_size=2, in_channels=16, depth=2, head_size=8, hidden=16, pos_embed_max=8, context_embed=16, freq_embed=8`), random weights, assert `forward((1, 16, 16, 16), (1,), (1, 2048), (1, 4, 16), None)` returns `(1, 16, 16, 16)`. Repeat with `skip_layers=Some(&[0])` and assert finite output.
- **`lazy_sd_samplers_sd3::flow_match_euler_sample`**: tiny MmDiT (as above), `num_inference_steps=2`, assert returned latent shape `(1, 16, H/8, W/8)` and finite values. Repeat with `slg_config=Some(...)` covering the window.
- **End-to-end binary**: published SD3-medium checkpoint, 256×256, 4 inference steps, fixed seed; assert output JPEG opens and pixel histogram is non-degenerate (not all-zero, not all-saturated).

## 6. Estimated scope

**5-6 sessions total**, breakdown:

- §1.1 lazy_sd3_text_encoder: 1 session.
- §1.2 lazy_sd3_vae (or extend lazy_sd_vae): 1 session.
- §1.3 lazy_mmdit extension: 1-2 sessions (split if MMDiT-X is in scope).
- §1.4 lazy_sd_samplers_sd3: 1 session.
- §3 binary revival + un-quarantine: 1 session.

If MMDiT-X is deferred to a follow-up (only SD3-medium and SD 3.5-large + SD 3.5-large-turbo work after this batch; SD 3.5-medium with `--use-slg` is the gated case), total is 5 sessions and SD 3.5-medium ships in a 6th.

## References

- Eager source (under `fuel-transformers/src/_models_retired/`):
  - `diffusion/mmdit/{model.rs, blocks.rs, embedding.rs, projections.rs}`
  - `diffusion/stable_diffusion/{vae.rs, clip.rs, unet_2d_blocks.rs (resnet/midblock helpers)}`
- Eager binary (recover from git history at `cfcb35cf~1:fuel-examples/examples/stable-diffusion-3/`):
  - `main.rs` (273 LOC), `clip.rs` (234 LOC), `vae.rs` (93 LOC), `sampling.rs` (84 LOC).
- Phase H retirement commit: `cfcb35cf` — `feat(retire): Phase H — eager fuel-transformers/models retired`.
- Already-shipped lazy modules:
  - `fuel-core/src/lazy_mmdit.rs` (substrate to extend).
  - `fuel-core/src/lazy_clip.rs`, `lazy_t5.rs` (text-encoder substrates).
  - `fuel-core/src/lazy_sd_vae.rs` (VAE substrate to extend or fork).
  - `fuel-core/src/lazy_sd_samplers_euler.rs`, `lazy_sd_samplers_unipc.rs` (sampler-file split convention).
  - `fuel-core/src/lazy_flux.rs` (sibling MMDiT-based pipeline; reuse `sampling::get_noise`).
- Sibling shipped session prompts (style + cross-reference):
  - `docs/session-prompts/shipped/port-mmdit.md`
  - `docs/session-prompts/shipped/port-sd-samplers.md`
  - `docs/session-prompts/shipped/port-flux.md`
- External:
  - SD3 paper (rectified flow): <https://arxiv.org/abs/2403.03206>.
  - SD3.5 reference impl: <https://github.com/Stability-AI/sd3.5>.
  - ComfyUI MMDiT-X impl: <https://github.com/comfyanonymous/ComfyUI/blob/78e133d0415784924cd2674e2ee48f3eeca8a2aa/comfy/ldm/modules/diffusionmodules/mmdit.py>.
  - SD3.5 SLG defaults: <https://github.com/Stability-AI/sd3.5/blob/4e484e05308d83fb77ae6f680028e6c313f9da54/sd3_infer.py#L388-L394>.
  - TAESD3 latent scale/shift constants (1.5305 / 0.0609): <https://github.com/comfyanonymous/ComfyUI/blob/3c60ecd7a83da43d694e26a77ca6b93106891251/nodes.py#L721-L723>.
