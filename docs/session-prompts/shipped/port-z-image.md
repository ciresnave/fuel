# Port: Z-Image (T2I diffusion)

## Eager source

- `fuel-transformers/src/models/diffusion/z_image/transformer.rs` (1101 LOC)
  — Main DiT transformer.
- `fuel-transformers/src/models/diffusion/z_image/vae.rs` (688 LOC)
  — VAE encoder + decoder.
- `fuel-transformers/src/models/diffusion/z_image/text_encoder.rs` (454 LOC)
  — Text encoder (likely T5 or CLIP variant; verify).
- `fuel-transformers/src/models/diffusion/z_image/scheduler.rs` (241 LOC)
  — Noise scheduler.
- `fuel-transformers/src/models/diffusion/z_image/preprocess.rs` (169 LOC)
  — Image preprocessing (resize, normalize, pad).
- `fuel-transformers/src/models/diffusion/z_image/sampling.rs` (133 LOC)
  — Sampling loop.
- `fuel-transformers/src/models/diffusion/z_image/mod.rs` (43 LOC)
  — Composition.

Total: ~2829 LOC across 7 files. Largest single diffusion port.

## Lazy module name

`fuel-core/src/lazy_z_image/{mod.rs, transformer.rs, vae.rs,
text_encoder.rs, scheduler.rs, preprocess.rs, sampling.rs}`
(directory module).

## Architecture summary

Z-Image is the largest Phase F port by LOC. Detailed architecture
needs investigation — spec is structured for an explore-first
session before code:

- DiT-based T2I generator (text → image).
- Custom VAE (688 LOC — bigger than SD VAE, so likely
  higher-resolution or more channels).
- Custom text encoder.
- Custom noise scheduler.

## Primitives needed

To be filled in during the explore-first sub-port (sub-port 0).
Likely overlap with port-mmdit.md / port-flux.md primitives:
modulation, RoPE patch positions, cross-attention.

## Reusable modules

Likely:

- `lazy_mmdit` — if the DiT shape matches MMDiT.
- `lazy_sd_vae` — VAE pattern.
- `lazy_clip` or `lazy_t5` — text encoder.

To be confirmed.

## Open questions

- **What is Z-Image?** No publicly recognized paper. Eager file
  was added in this project — read the eager files end-to-end
  before drafting code. Likely an in-house experimental T2I
  generator.
- Does it share substrate with MMDiT or Flux?
- What's the latent space (channels, downsample)?
- What's the noise schedule?

## Splits

Mandatory split given the size:

0. **Sub-port 0 (explore)**: read all 7 eager files end-to-end,
   fill in the architecture summary, primitives, and reusable
   modules sections of this spec. Output: an updated spec ready
   for sub-ports 1–7. This sub-port is *spec work*, not code, but
   it's a discrete prompt.
1. **Sub-port 1**: `preprocess.rs` + `scheduler.rs` — pure
   host-side helpers. Standalone tests.
2. **Sub-port 2**: `text_encoder.rs` — depending on whether it's
   T5 / CLIP / custom.
3. **Sub-port 3**: `vae.rs` — VAE encode + decode.
4. **Sub-port 4**: `transformer.rs` — main DiT. Largest sub-port.
   May further split if it doesn't fit a session.
5. **Sub-port 5**: `sampling.rs` + `mod.rs` — composition +
   end-to-end.

## Test strategy

- Tiny configs throughout.
- Pure host preprocessing → golden tests.
- VAE round-trip.
- End-to-end: text → image, finite, expected shape.

## References

- Eager source: `fuel-transformers/src/models/diffusion/z_image/*`
- (Paper / repo references to be added by sub-port 0.)
- Likely siblings: `lazy_mmdit`, `lazy_sd_vae`, `lazy_clip`,
  `lazy_t5`.
