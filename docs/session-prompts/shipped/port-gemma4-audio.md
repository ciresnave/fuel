# Port: Gemma4 audio (Conformer)

## Eager source

- `fuel-transformers/src/models/llm/gemma4/audio.rs` (874 LOC)
  — Gemma4 audio encoder: Subsampling Conv1d Projection (SSCP) +
    Conformer blocks (feed-forward → multi-head self-attention with
    chunked attention + rel-pos bias → lightweight Conv1d → ff →
    LayerNorm).

## Lazy module name

`fuel-core/src/lazy_gemma4_audio.rs` (new file). Mirror the
existing `lazy_gemma4_text` / `lazy_gemma4_vision` /
`lazy_gemma4_mm_embed` split.

## Architecture summary

Conformer is the standard audio encoder pattern from ASR. Each
Conformer block:

1. Half-step feed-forward (residual = 0.5 * x_prev + FF(x_prev))
2. Multi-head self-attention with:
   - **Chunked attention** — divides the sequence into fixed-size
     chunks (e.g. 128 frames) and computes attention within each
     chunk + a small "left context" window of prior chunks. Used
     to bound compute for long audio.
   - **Relative position bias** — Shaw-style or T5-style
     positional encoding rather than RoPE.
3. Lightweight Conv1d (kernel=5 or so, depthwise + pointwise, with
   GLU gating).
4. Half-step feed-forward.
5. LayerNorm at the end.

SSCP front-end: two strided Conv2d layers acting on
`(B, 1, T, n_mels)` to subsample temporally and project to the
Conformer model dim.

## Primitives needed

- **Chunked attention with left-context window** — build the
  block-diagonal-plus-left-band attention mask from chunk_size and
  left_chunks parameters, feed into existing SDPA. Host-built
  mask, `const_*_like`-emitted at forward time.
- **Relative position bias table** — Shaw embedding table, indexed
  by `(i - j)` offsets clipped to a window. Standard pattern;
  emit as a lookup using `index_select` from a learned
  `(2*max_rel + 1, num_heads)` table.
- **GLU activation** — already shipped (`lazy_*` modules use it).
- **Depthwise Conv1d** — `LazyTensor::conv1d` with
  `groups = channels`. Verify the surface supports it; if it's
  Conv2d-only with `groups`, route through Conv2d with a unit H
  dim and squeeze back.

## Reusable modules

- `lazy_gemma4_text`, `lazy_gemma4_vision`, `lazy_gemma4_mm_embed`
  — sibling modules already exist; follow their patterns for the
  shared types (Gemma4Config audio sub-config, RMSNorm-no-scale
  pattern).
- `lazy_whisper_audio` (after port-whisper-audio.md ships) — feeds
  mel input.

## Open questions

- Chunked attention: does the eager file use a left-context
  window, or pure per-chunk independence? Read the eager
  `forward_attention` carefully and document the actual
  mask shape.
- Rel-pos clipping window: how large? Drives the
  `(2*max_rel+1, num_heads)` table size.
- Audio preprocessing — does Gemma4 audio expect log-mel
  filterbank (Whisper-style 80 bands), or raw waveform with the
  SSCP front-end handling everything? Check the input dtype/shape
  the eager `audio_features_forward` accepts.
- How does Gemma4 audio fuse into the multimodal embedder
  (`lazy_gemma4_mm_embed`)? Image is shipped — audio needs an
  analogous projector + slot-scatter. Cross-check with
  `lazy_gemma4_mm_embed::forward_audio` if it exists, or add it
  as part of this port.

## Splits

Recommended split:

1. **Sub-port 1**: SSCP front-end + Conformer block with chunked
   attention + rel-pos bias. Tested standalone on tiny config.
2. **Sub-port 2**: audio-side wiring into `lazy_gemma4_mm_embed`
   (projector + slot-scatter contract with the text LM). May need
   eager-side audit to see how the multimodal composition is
   shaped.

## Test strategy

- Tiny config: T_in=64 frames, n_mels=8, model_dim=16, depth=2,
  num_heads=4, chunk_size=16, left_chunks=1.
- Verify output shape `(B, T_in / subsample_factor, model_dim)`
  and finite values.
- Chunked attention mask golden test: build the mask manually for
  a 4-chunk sequence with left_chunks=1, assert it matches the
  built mask exactly.
- Rel-pos bias lookup golden test: hand-compute a tiny (i, j)
  offset matrix, assert lookup matches.

## References

- Eager source: `fuel-transformers/src/models/llm/gemma4/audio.rs`
- Gemma 4 technical report (Google) — audio section.
- Conformer paper: <https://arxiv.org/abs/2005.08100>
- Already-shipped Gemma4 family: `lazy_gemma4_text`,
  `lazy_gemma4_vision`, `lazy_gemma4_mm_embed`.
