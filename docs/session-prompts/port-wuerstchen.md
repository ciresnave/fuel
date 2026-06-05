# Port: Wuerstchen (cascaded diffusion)

## Eager source

- `fuel-transformers/src/models/diffusion/wuerstchen/diffnext.rs` (402 LOC)
  — Wuerstchen v2 Stage B / Stage C "DiffNext" diffusion UNet.
- `fuel-transformers/src/models/diffusion/wuerstchen/paella_vq.rs` (217 LOC)
  — Paella-VQ tokenizer (VQ-VAE for Stage A latent space).
- `fuel-transformers/src/models/diffusion/wuerstchen/prior.rs` (107 LOC)
  — Prior model: text → low-resolution latent.
- `fuel-transformers/src/models/diffusion/wuerstchen/ddpm.rs` (109 LOC)
  — DDPM scheduler.
- `fuel-transformers/src/models/diffusion/wuerstchen/attention_processor.rs` (121 LOC)
  — Cross-attention block.
- `fuel-transformers/src/models/diffusion/wuerstchen/common.rs` (218 LOC)
  — Shared layer helpers (ResBlock + GlobalResponseNorm + AttnBlock).
- `fuel-transformers/src/models/diffusion/wuerstchen/mod.rs` — composition.

Total: ~1176 LOC across 7 files.

## Lazy module name

`fuel-core/src/lazy_wuerstchen/{mod.rs, common.rs, attention_processor.rs,
prior.rs, diffnext.rs, paella_vq.rs, ddpm.rs}` (directory module).

## Architecture summary

Wuerstchen v2 is a cascaded latent diffusion model from
Stability AI:

- **Stage A**: PaellaVQ tokenizer compresses image to discrete VQ
  latents (~24x compression vs raw pixels).
- **Stage B**: DiffNext UNet diffuses on the VQ latents
  conditioned on Stage C output.
- **Stage C** ("Prior"): smaller diffusion model that maps text
  embedding → low-resolution latent map. The text→C→B→A cascade
  is the key insight; each stage works in a smaller latent space.

DiffNext UNet:
- Residual blocks with GlobalResponseNorm (ConvNeXt v2 style).
- Cross-attention to text at certain depths.
- Time + class embedding.

PaellaVQ:
- Encoder: 5 strided conv blocks.
- VQ codebook: discrete embedding lookup.
- Decoder: 5 transposed conv blocks.

## Primitives needed

- **GlobalResponseNorm** — `gamma * (x / x.norm(dim=spatial)) + beta`.
  Standard ConvNeXt v2 primitive. Reusable as
  `lazy_wuerstchen::common::global_response_norm`.
- **VQ codebook lookup** — `LazyTensor::index_select` against
  the codebook matrix, plus a nearest-neighbor lookup for the
  encoder side (argmin distance). The argmin path is host-side
  during inference if codebook is fixed; runtime VQ during
  inference is rare.
- Cross-attention block — standard.

## Reusable modules

- `lazy_convnext` — already has GlobalResponseNorm; lift if
  feasible.
- `lazy_sd_unet` — cross-attention block pattern.
- `lazy_clip` — text encoder upstream.

## Open questions

- Stage A VQ encoder: is it actually used at inference, or only
  the decoder? Stable Cascade (commercial Wuerstchen v3) only
  ships the decoder for inference. Confirm by reading
  `mod.rs::generate` or equivalent.
- Stage C prior input shape: text embed + timestep + noise →
  small 2D map. Confirm dims.
- Scheduler: DDPM is the eager file's choice. Match exactly; if
  alternative schedulers are needed (DDIM, etc.), they live in
  port-sd-samplers.md and we cross-link.

## Splits

Recommended split:

1. **Sub-port 1**: `common.rs` + `attention_processor.rs` — shared
   primitives. Standalone tests.
2. **Sub-port 2**: `paella_vq.rs` — VQ-VAE decoder (Stage A
   decoder for inference). Standalone round-trip test on tiny
   config.
3. **Sub-port 3**: `prior.rs` + `ddpm.rs` — Stage C prior + DDPM
   scheduler. Standalone tiny-step sampling test.
4. **Sub-port 4**: `diffnext.rs` — Stage B DiffNext UNet.
   Standalone tiny test.
5. **Sub-port 5**: `mod.rs` — composition (text → C → B → A
   → image). End-to-end tiny integration test.

## Test strategy

- Tiny configs throughout; spatial dims 8x8 for the largest
  stage, 2x2 for the smallest.
- GlobalResponseNorm golden test.
- VQ decoder shape check.
- Sampling smoke test: 2 DDPM steps, finite output.
- End-to-end: text → image of shape `(1, 3, 32, 32)`, finite,
  values in [-1, 1] (after the final activation).

## References

- Eager source: `fuel-transformers/src/models/diffusion/wuerstchen/*`
- Wuerstchen paper: <https://arxiv.org/abs/2306.00637>
- Stable Cascade (v3): <https://stability.ai/news/introducing-stable-cascade>
- Reference impl: <https://github.com/dome272/Wuerstchen>
- Already-shipped: `lazy_convnext`, `lazy_sd_unet`, `lazy_clip`.
