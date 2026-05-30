# Session prompt — Mimi xn-parity (StreamingModule mask extension)

## What this session is for

Fuel already has a Mimi implementation
(`fuel-transformers/src/models/audio/mimi/`, ~2.7k LOC) ported
from an earlier Candle snapshot. xn's current Mimi
(`xn-core/src/models/mimi.rs`, ~2.1k LOC consolidated into one
file) has diverged from what Fuel carries forward — most
significantly, **xn threads a `StreamMask` parameter through every
`StreamingModule::step` call**.

Fuel today:
```rust
pub trait StreamingModule {
    fn step(&mut self, xs: &StreamTensor) -> Result<StreamTensor>;
    fn reset_state(&mut self);
}
```

xn:
```rust
pub trait StreamingModule<T: WithDTypeF, B: Backend> {
    fn step(&mut self, xs: &StreamTensor<T, B>, mask: &StreamMask)
        -> Result<StreamTensor<T, B>>;
}
```

The mask threads through 12+ `impl StreamingModule` sites in
xn's mimi.rs and is consumed by the downsample/upsample/resnet
blocks to gate per-batch-element state updates. Without it,
batched streaming where individual sequences finish at different
times can't preserve state for finished elements while continuing
to update active ones.

This session makes Fuel's `StreamingModule` mask-aware and
updates the 9 existing impls to thread the mask through. End
state: Fuel's Mimi has feature parity with xn's mimi w.r.t. the
streaming-mask pattern.

## Why this matters

The 2026-05-29 session ported xn's `StreamMask` +
`apply_state_mask` into `fuel-core/src/streaming.rs` as additive
new types. But the existing `StreamingModule` trait doesn't
accept a mask, so the new types can't be used from Mimi's
streaming-decode loop without this refactor.

This isn't speculative — it's the direct blocker between the
streaming primitives Fuel just shipped and actually using them
in models.

## The refactor

### Step 1: Extend the trait

`fuel-core/src/streaming.rs::StreamingModule`:

```rust
pub trait StreamingModule {
    /// Process the next chunk of input and return any output that
    /// is ready.
    ///
    /// `mask` is the per-batch-element active mask: rows where
    /// `mask.is_active(i)` is `true` update their state from `xs`;
    /// finished rows preserve existing state. Pass
    /// `&StreamMask::empty()` when batching isn't needed (treats
    /// all rows as active).
    fn step(&mut self, xs: &StreamTensor, mask: &StreamMask)
        -> Result<StreamTensor>;

    fn reset_state(&mut self);
}
```

### Step 2: Update existing impls

Investigated 2026-05-29 — exactly 9 impls in the codebase:

```
fuel-core/src/streaming.rs:355   impl<T: Module> StreamingModule for Map<T>
fuel-transformers/src/models/audio/mimi/conv.rs:326   StreamableConv1d
fuel-transformers/src/models/audio/mimi/conv.rs:424   StreamableConvTranspose1d
fuel-transformers/src/models/audio/mimi/conv.rs:505   ConvDownsample1d
fuel-transformers/src/models/audio/mimi/conv.rs:554   ConvTrUpsample1d
fuel-transformers/src/models/audio/mimi/seanet.rs:124  SeaNetResnetBlock
fuel-transformers/src/models/audio/mimi/seanet.rs:286  SeaNetEncoder
fuel-transformers/src/models/audio/mimi/seanet.rs:451  SeaNetDecoder
fuel-transformers/src/models/audio/mimi/transformer.rs:688  StreamingTransformer
fuel-transformers/src/models/audio/mimi/transformer.rs:770  ProjectedTransformer
```

Per impl: add `mask: &StreamMask` param; pass `mask` through to
inner module step calls; use `apply_state_mask` where there's
internal state being updated (per the xn implementations — see
ConvDownsample1d and ConvTrUpsample1d for the canonical pattern).

### Step 3: Update callers of `.step()`

Most callers pass through within `StreamingModule` impls (handled
by step 2). Top-level callers in tests / examples / inference
loops pass `&StreamMask::empty()` for the no-batching case.

### Step 4: Cross-reference xn's per-block mask usage

For each Fuel impl, look at the corresponding xn impl in
`https://github.com/LaurentMazare/xn/blob/main/xn-core/src/models/mimi.rs`
and check whether the mask is just threaded (ignored) or actually
consumed. Most are threaded. The ones that consume the mask are
the downsample/upsample modules where state buffers need
per-row gating during accumulation.

## Other xn-vs-Fuel Mimi divergences (lower priority)

While auditing, also note for future sessions:

- xn organizes Mimi as one 2.1k-LOC file; Fuel splits it across 6
  files. Different style choice; neither is wrong; no action.
- xn's `KvCache` is mimi-local; Fuel uses
  `fuel_core::kv_cache::*`. Already handled by re-export.
- xn passes `Backend` as a type parameter throughout; Fuel uses
  dtype-erased `Tensor`. This is the broader architectural
  difference captured in the 2026-05-29 xn audit memo.

## Out of scope

- Moshi port (depends on this session landing first; sits on
  Mimi).
- Demucs port (independent of Mimi, can ship in parallel
  session).
- The streaming-training extension to `StreamingModule` (autograd-
  aware variant for chunked BPTT). Captured in
  [`project_deferred_streaming_training`](../../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_deferred_streaming_training.md);
  not this session.

## Test scope

- All existing Mimi tests pass with the new trait signature
  (most tests pass `&StreamMask::empty()` as a one-line change).
- Add a focused test for the downsample/upsample mask-gating
  behavior: build a 2-row batch where row 1 is masked off
  partway through, verify row 1's state is preserved while row 0
  continues to accumulate.

## Scope realism

Mechanical refactor across 9 impl sites + 1 trait change + some
callers. ~1-2 sessions. Most of the work is straightforward
mask-threading; the consumed-not-just-threaded sites (downsample,
upsample) need careful translation from xn's reference impl.

Link: [`fuel-core/src/streaming.rs`](../../fuel-core/src/streaming.rs)
for the existing primitives this session extends.
