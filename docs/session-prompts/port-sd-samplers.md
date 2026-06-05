# Port: Stable Diffusion samplers + attention building blocks

## Eager source

- `fuel-transformers/src/models/diffusion/stable_diffusion/attention.rs` (576 LOC)
  — Cross-attention / self-attention building blocks for SD UNet.
- `fuel-transformers/src/models/diffusion/stable_diffusion/ddim.rs` (208 LOC)
  — DDIM scheduler.
- `fuel-transformers/src/models/diffusion/stable_diffusion/ddpm.rs` (208 LOC)
  — DDPM scheduler.
- `fuel-transformers/src/models/diffusion/stable_diffusion/uni_pc.rs` (1017 LOC)
  — UniPC scheduler (largest single sampler).
- `fuel-transformers/src/models/diffusion/stable_diffusion/euler_ancestral_discrete.rs` (230 LOC)
  — Euler-ancestral scheduler.
- `fuel-transformers/src/models/diffusion/stable_diffusion/schedulers.rs` (75 LOC)
  — Common scheduler trait + helpers.

Total: ~2294 LOC across 6 files.

## Lazy module name

- `fuel-core/src/lazy_sd_samplers.rs` (DDIM / DDPM / Euler /
  UniPC + Scheduler trait, all host-side scalar control).
- `fuel-core/src/lazy_sd_attention.rs` (CrossAttention /
  SelfAttention building blocks used inside the UNet — though
  `lazy_sd_unet` already exists; verify whether it inlined these
  blocks or imports from a separate file).

## Architecture summary

**Schedulers** are pure host-side scalar control: given a noise
sample, a model-predicted noise/velocity, and timestep state, they
produce the next sample. No graph ops. Each scheduler implements:

```rust
trait Scheduler {
    fn set_timesteps(&mut self, num_inference_steps: usize);
    fn step(&mut self, model_output: &LazyTensor, timestep: usize,
            sample: &LazyTensor) -> Result<LazyTensor>;
    fn add_noise(&self, original: &LazyTensor, noise: &LazyTensor,
                 timesteps: &[usize]) -> Result<LazyTensor>;
}
```

The `step` implementation mixes scalar host f64 arithmetic
(scheduler coefficients) with tensor ops (sample updates).
Coefficients are emitted as `const_f32_like` or `Scalar`.

**Attention building blocks**: standard cross-attention used by
the SD UNet at certain depths. CrossAttention(Q from latent,
K/V from text embed). Already partially implemented inside
`lazy_sd_unet` — port the standalone version to clean things up
or leave as-is if the inlined version is the only consumer.

## Primitives needed

- None new on graph side — pure host control flow + existing
  binary ops.

## Reusable modules

- `lazy_sd_unet` — consumer of attention + schedulers.
- `lazy_sd_vae`, `lazy_sd_text_encoder` — pipeline siblings.

## Open questions

- The eager `schedulers.rs` defines a trait. Do we want the lazy
  port to ship as a single enum or as separate types behind a
  trait? Preference: each scheduler is its own type; pipelines
  generic over `S: SdScheduler`. Avoids dyn dispatch and is
  cleaner ergonomically.
- UniPC is the largest. Worth its own sub-port.
- Does `lazy_sd_unet` already inline an attention block, or does
  it import from a not-yet-ported module? Check before deciding
  whether the attention port is needed.

## Splits

Recommended split:

1. **Sub-port 1**: `lazy_sd_attention.rs` if `lazy_sd_unet` does
   *not* already inline the blocks. Otherwise skip and document
   that the attention surface was already shipped inside
   `lazy_sd_unet`.
2. **Sub-port 2**: Scheduler trait + DDIM + DDPM (small,
   straightforward).
3. **Sub-port 3**: Euler-ancestral (~230 LOC).
4. **Sub-port 4**: UniPC (~1000 LOC; biggest single sampler;
   careful porting against the eager file).

## Test strategy

- Schedulers: golden tests against the eager scheduler's output
  for a fixed seed (hand a Vec<f32> sample + Vec<f32> noise +
  scheduler config, assert `step` produces matching f32 output).
- Attention: shape + finite check; matmul against a known small
  weight set.

## References

- Eager source: `fuel-transformers/src/models/diffusion/stable_diffusion/*`
- diffusers reference: <https://github.com/huggingface/diffusers>
  schedulers.
- DDIM: <https://arxiv.org/abs/2010.02502>
- DDPM: <https://arxiv.org/abs/2006.11239>
- UniPC: <https://arxiv.org/abs/2302.04867>
- Already-shipped: `lazy_sd_unet`, `lazy_sd_vae`,
  `lazy_sd_text_encoder`.
