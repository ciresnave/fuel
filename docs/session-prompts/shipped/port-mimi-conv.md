# Port: Mimi conv (streaming/causal Conv1d primitive)

## Eager source

- `fuel-transformers/src/models/audio/mimi/conv.rs` (688 LOC)
  — Streaming-aware Conv1d / ConvTranspose1d / pooling helpers.
    Several variants:
    - `StreamableConv1d` — causal Conv1d with weight-norm + persistent
      ring buffer for streaming.
    - `StreamableConvTranspose1d` — output-overlap-add streaming.
    - `ConvDownsample1d` / `ConvUpsample1d` — strided wrappers.
    - `NormConv1d` — choice of WeightNorm / TimeGroupNorm / Identity.

## Lazy module name

`fuel-core/src/lazy_mimi_conv.rs` (new file). Exports the same
public surface as the eager file, but with `LazyTensor` instead of
`Tensor` and **explicit ring-buffer state** carried as a
`StreamConv1dState` value passed through `step(state, x) -> (state, y)`
rather than embedded in `&mut self`.

## Architecture summary

Mimi is Kyutai's neural audio codec — encoder reads PCM and
streams compressed tokens; decoder streams the inverse. The conv
layers must be **causal** (output[t] depends only on input[<=t])
and **streaming-friendly** (process arbitrarily small chunks while
preserving state across calls). Concretely:

- Each conv input is left-padded with `(kernel - 1) * dilation`
  zeros at sequence start, then trimmed to maintain
  `output_len == input_len // stride`.
- Streaming: a ring buffer of the last `(kernel - 1) * dilation`
  input samples persists between `step` calls so the next chunk
  sees the correct historical context.
- ConvTranspose1d: outputs overlap the next chunk, so a similar
  buffer is kept on the output side.

The lazy port follows the project's stateless-by-default
philosophy:

```rust
pub struct StreamConv1dState {
    pub buf: Option<LazyTensor>,  // last (k-1)*d samples, shape (B, C, L_buf)
}
pub fn streamable_conv1d_step(
    state: StreamConv1dState,
    x: &LazyTensor,           // (B, C_in, L_chunk)
    weight: &LazyTensor,
    bias: Option<&LazyTensor>,
    cfg: &StreamableConv1dConfig,
) -> Result<(StreamConv1dState, LazyTensor)>;
```

One-shot (non-streaming) call is just `step(empty_state, full_x)
.map(|(_, y)| y)`.

## Primitives needed

- `LazyTensor::conv1d` — should already exist (used by lazy_mamba
  via `causal_conv1d`-fused-op path). Verify it's exposed; if not,
  thread through `Op::Conv1D` to the existing eager kernel.
- WeightNorm reparameterization. Two options:
  1. Bake WN at load time: weights come in as `(g, v)` and we emit
     `w = g * v / ||v||` once, ship a plain Conv1d. Simpler. This
     is what `lazy_mimi_seanet` already does for SEANet conv layers.
  2. Wrap as a graph op so weights stay reparameterized. Only
     needed for training. Defer until Phase G.
  Choose option 1.

## Reusable modules

- `lazy_mimi_seanet` — already does WeightNorm baking. Borrow the
  helper.
- `lazy_mimi_transformer` — pattern for state-as-value streaming
  (StreamingMHA carries `KvState` similarly).
- `lazy_mamba::apply_causal_conv1d` — already a fused-op call for
  the simple causal Conv1d case. Reusable as the inner kernel after
  left-padding.
- `LazyTensor::pad_with_zeros` and `narrow` for the ring buffer
  concat/slice.

## Open questions

- TimeGroupNorm — the eager file has it. Is there a downstream
  consumer that uses Norm::TimeGroupNorm? Check
  `lazy_mimi_seanet` and `audio/mimi/encodec.rs` — probably yes for
  some Mimi presets. If used, port it; if not, leave the variant
  out and `panic!`-error on its config rather than silently fall
  back.
- Dilation > 1: do any Mimi presets use it? If yes, ring-buffer
  size becomes `(k - 1) * d` instead of `k - 1`. Make the
  implementation handle dilation from the start.
- ConvTranspose1d streaming output-overlap-add — does the lazy
  graph support an in-place add on a sliding window? If not, just
  realize the slice each step and let it materialize; we'll
  measure and decide if it needs a fused op later.

## Splits

This is large (~700 LOC eager) and touches a new state-carrying
pattern. Recommended split:

1. **Sub-port 1**: `StreamableConv1d` causal + ring-buffer state.
   Weight-norm baked at load. No ConvTranspose, no Norm variants
   beyond Identity + WeightNorm-baked.
2. **Sub-port 2**: `StreamableConvTranspose1d` + output ring buffer.
3. **Sub-port 3**: `ConvDownsample1d` / `ConvUpsample1d` wrappers
   (thin — should be ~30 lines on top of 1+2).
4. **Sub-port 4**: TimeGroupNorm if needed by encodec consumers.

Each sub-port is its own session and its own commit. After 1+2
ship, port-mimi-encodec.md can start in parallel against sub-port
3 / 4 if needed.

## Test strategy

- **One-shot ≡ streaming chunk-wise equivalence**: feed a length-128
  signal through `streamable_conv1d_step(empty, full)` and the same
  signal in chunks of size 1, 8, 32, asserting the concatenated
  streaming output matches the one-shot output bit-for-bit.
- **Causality**: zero the last half of the input, assert the first
  half of the output is unchanged.
- **WeightNorm baking**: golden test against a tiny known weight
  pair `(g, v)`.
- Real Mimi weights aren't needed for unit tests; the integration
  test lands in `lazy_mimi_encodec` once that ships.

## References

- Eager source: `fuel-transformers/src/models/audio/mimi/conv.rs`
- Mimi paper: <https://arxiv.org/abs/2410.00037> (Moshi)
- Reference impl: <https://github.com/kyutai-labs/moshi>
- Already-shipped: `lazy_mimi_seanet`, `lazy_mimi_transformer`,
  `lazy_mamba::apply_causal_conv1d`.
- Feedback memory: streaming patterns shipped 2026-05-29 (see
  `project_overnight_session_2026_05_29` for the
  `StreamMask` / `apply_state_mask` precedent).
