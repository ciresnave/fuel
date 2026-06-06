# Port: Flux (model + autoencoder + sampling + quantized)

## Eager source

- `fuel-transformers/src/models/diffusion/flux/model.rs` (639 LOC)
  — Flux DiT: DoubleStreamBlock + SingleStreamBlock variants
    specialized for Flux (qk_norm, parallel attention shape, etc.).
- `fuel-transformers/src/models/diffusion/flux/autoencoder.rs` (453 LOC)
  — VAE for Flux (different from SD VAE's latent shape).
- `fuel-transformers/src/models/diffusion/flux/quantized_model.rs` (473 LOC)
  — GGUF-quantized Flux model.
- `fuel-transformers/src/models/diffusion/flux/sampling.rs` (124 LOC)
  — Flow-matching scheduler + sampling loop.

## Lazy module name

`fuel-core/src/lazy_flux/{mod.rs, autoencoder.rs, quantized.rs,
sampling.rs}` (directory module). Mirrors the eager split.

## Architecture summary

Flux is BlackForestLabs' MMDiT-based image diffusion. Components:

1. **Flux model**: MMDiT-style DiT (text+image stream blocks)
   with Flux-specific changes:
   - QK-Norm: RMSNorm applied to Q and K independently inside
     attention.
   - Parallel attention + MLP: SingleStream block computes
     attention and MLP from a shared projection, summed at output.
   - Modulation: 9 params per block (vs SD3's 6).
2. **Autoencoder**: VAE encoder for input image → latents,
   decoder for output latents → image. 16-channel latent space,
   8x downsampling.
3. **Flow-matching scheduler**: linear (or shifted-linear) noise
   schedule between t=0 (data) and t=1 (noise). Sampling steps
   integrate the velocity field predicted by the DiT.
4. **Quantized variant**: GGUF Q4_0 / Q4_K_M / Q5_0 / Q8_0
   weights loaded into a GGUF-backed model. Same forward; weights
   come from a different loader.

## Primitives needed

- **QK-Norm RMSNorm-inside-attention** — apply RMSNorm to Q and K
  before SDPA. Just wrap existing `LazyTensor::rms_norm`.
- **Flow-matching scheduler** — pure host-side scalar control
  loop. No new graph ops.
- **Q-matmul GGUF** — already exists (`Nf4Matmul` + Q4_K family
  shipped). Verify the Flux GGUF loader can use the existing
  fused-op surface.

## Reusable modules

- `lazy_mmdit` — substrate for the DiT blocks. Flux specializations
  are deltas on top.
- `lazy_sd_vae` — VAE pattern (though latent channels differ).
- `lazy_clip` + `lazy_t5` — text encoders feeding Flux.
- `Nf4Matmul` + `lazy.rs` Q-matmul helpers — for the quantized
  variant.

## Open questions

- Does the Flux DiT diverge from `lazy_mmdit` enough to deserve
  its own block implementation, or can it parameterize over
  `lazy_mmdit::DoubleStreamBlock`? Need to read `model.rs` after
  port-mmdit.md ships.
- VAE: SD VAE has 4 latent channels, Flux has 16. Different scale
  factor for residual blocks. May not be reusable from
  `lazy_sd_vae` — likely a fresh port.
- Sampling: which schedulers does the eager file support? Flow-
  matching linear is standard; are there alternatives we need to
  carry?

## Splits

Recommended split (gates on port-mmdit.md):

1. **Sub-port 1** (depends on mmdit): `lazy_flux/mod.rs` — Flux
   DiT with QK-Norm + 9-param modulation. Standalone test.
2. **Sub-port 2**: `lazy_flux/autoencoder.rs` — Flux VAE encode
   + decode. Standalone test.
3. **Sub-port 3**: `lazy_flux/sampling.rs` — flow-matching
   scheduler + sampling loop. Integration test that runs a tiny
   model for 2 steps.
4. **Sub-port 4**: `lazy_flux/quantized.rs` — GGUF loader →
   `LazyFluxModel`. Reuses sub-port 1's forward.

## Test strategy

- Tiny DiT: hidden=16, num_heads=4, depth=2, seq_text=8,
  seq_image=16. Verify output shape and finite values.
- VAE round-trip: encode + decode a `(1, 3, 64, 64)` image, assert
  decoded shape matches input.
- Flow-matching scheduler: 2-step sampling on tiny config, assert
  output finite + bounded.

## References

- Eager source: `fuel-transformers/src/models/diffusion/flux/*`
- Black Forest Labs: <https://blackforestlabs.ai/>
- Reference impl: <https://github.com/black-forest-labs/flux>
- Flow-matching paper: <https://arxiv.org/abs/2210.02747>
- Sibling: `lazy_mmdit` (substrate), `lazy_sd_vae` (VAE pattern),
  `Nf4Matmul` (quantized).
