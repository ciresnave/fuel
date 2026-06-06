# Port: Mimi encodec (top-level codec composition)

## Eager source

- `fuel-transformers/src/models/audio/mimi/encodec.rs` (272 LOC)
  — Top-level Mimi codec orchestrator. Composes SEANet conv
    backbone + Mimi transformer + Mimi resampler + Mimi quantization
    into encode(pcm) → tokens, decode(tokens) → pcm.

## Lazy module name

Extend `fuel-core/src/lazy_mimi.rs` (already exists per `lib.rs`
inspection — it carries the `MimiModel` composition; the
sub-modules `lazy_mimi_seanet`, `lazy_mimi_transformer`,
`lazy_mimi_resampler`, `lazy_mimi_quantization` are already shipped).
This port wraps them under the encodec interface that's still
eager.

## Architecture summary

Mimi's encode pipeline:
1. Resample PCM 24 kHz → 12.5 Hz (host-side).
2. SEANet encoder Conv1d stack (`lazy_mimi_seanet`).
3. Mimi transformer encoder (`lazy_mimi_transformer`).
4. Mimi resampler (`lazy_mimi_resampler`).
5. RVQ quantization (`lazy_mimi_quantization`).

Decode pipeline is the mirror image.

This port is a *composition* port — most of the heavy lifting was
shipped earlier in the sub-modules. The work here is:
- Top-level `MimiCodec::encode(pcm)` / `decode(tokens)` API.
- Streaming variant: `MimiCodecStreamState` + `encode_step` /
  `decode_step`.

## Primitives needed

- All shipped: `lazy_mimi_seanet`, `lazy_mimi_transformer`,
  `lazy_mimi_resampler`, `lazy_mimi_quantization`.
- Gate: requires port-mimi-conv.md to ship first (SEANet might
  already inline a Conv1d that needs to be swapped for the
  streamable variant, depending on how `lazy_mimi_seanet` was
  written; check).

## Reusable modules

- All four `lazy_mimi_*` sub-modules.
- `lazy.rs::MimiModel` if it exists at this level.

## Open questions

- Does `lazy_mimi.rs` already contain the `MimiModel` top-level
  composition, or just the sub-module re-export? Read it first.
- Streaming vs one-shot — both required, or does the consumer
  only care about one mode? Real Moshi inference requires
  streaming.

## Splits

Single session if port-mimi-conv.md is shipped. If the streamable
state needs significant rework of the existing sub-modules, split
out the streamable refactor as a dedicated sub-port.

## Test strategy

- Tiny PCM (length 1024 samples) round-trip: encode → decode,
  compare to eager reference within a generous tolerance (lossy
  codec).
- Streaming equivalence: encode the same PCM in chunks of 256
  samples vs one-shot, assert the resulting tokens match.
- Decoded waveform shape matches expected sample count.

## References

- Eager source: `fuel-transformers/src/models/audio/mimi/encodec.rs`
- Mimi/Moshi paper: <https://arxiv.org/abs/2410.00037>
- Reference impl: <https://github.com/kyutai-labs/moshi>
- Already-shipped sub-modules: `lazy_mimi_seanet`,
  `lazy_mimi_transformer`, `lazy_mimi_resampler`,
  `lazy_mimi_quantization`, `lazy_mimi`.
