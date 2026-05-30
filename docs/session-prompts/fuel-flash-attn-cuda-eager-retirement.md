# Retire the eager `fuel-flash-attn-cuda{,-sys}` crates

## State entering this session

- **Lazy `Op::FlashAttn` path** — migrated to baracuda
  `fa2_sdpa_*_run_v2` in commit e359b041. Live oracle
  (`fuel-core/tests/flash_attn_cuda.rs`) passes F16 basic + causal on
  RTX 4070. baracuda alpha.60 vendors the full head_dim set
  {32, 64, 96, 128, 160, 192, 224, 256, 512} for FW.
- **Eager `fuel-flash-attn-cuda::flash_attn(Tensor, ...) -> Tensor`** —
  STILL calls `fuel_flash_attn_cuda_sys::run_mha` under the hood
  (986 LOC in `fuel-flash-attn-cuda/src/lib.rs`). Used by ~14
  transformer model files behind their own `#[cfg(feature = "flash-attn")]`
  gates.
- **`flash-attn` Cargo feature** — kept alive as transitional: now
  enables BOTH the migrated lazy launcher (no nvcc cost) AND the
  unmigrated eager wrapper (full nvcc build of vendored FA2 sources).

## Consumer inventory (the ~14 transformer files)

All call `fuel_flash_attn_cuda::flash_attn(q, k, v, scale, causal)`
behind `#[cfg(feature = "flash-attn")]`:

- `fuel-transformers/src/models/llm/llama.rs`
- `fuel-transformers/src/models/llm/granite.rs`
- `fuel-transformers/src/models/llm/granitemoehybrid.rs`
- `fuel-transformers/src/models/llm/helium.rs`
- `fuel-transformers/src/models/llm/gemma3.rs`
- `fuel-transformers/src/models/llm/gemma4/text.rs`
- `fuel-transformers/src/models/quantized/quantized_phi3.rs`
- `fuel-transformers/src/models/multimodal/voxtral/voxtral_llama.rs`
- `fuel-transformers/src/models/audio/mimi/transformer.rs`
- `fuel-transformers/src/models/diffusion/z_image/transformer.rs`
- `fuel-transformers/src/models/diffusion/wuerstchen/attention_processor.rs`
- `fuel-transformers/src/models/diffusion/stable_diffusion/{mod,attention}.rs`
- `fuel-transformers/src/models/diffusion/mmdit/blocks.rs`

Pattern is identical across them:

```rust
#[cfg(feature = "flash-attn")]
fn flash_attn(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool)
    -> Result<Tensor> {
    fuel_flash_attn_cuda::flash_attn(q, k, v, scale, causal)
}
#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool)
    -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}
```

## Two retirement paths

### Option A: rewrite the eager wrapper to call baracuda directly

Replace `cuda_fwd_t` inside `fuel-flash-attn-cuda/src/lib.rs` (which
currently dispatches through `fuel_flash_attn_cuda_sys::ffi::run_mha`)
to call `baracuda_kernels_fa2_sdpa_{f16,bf16}_run_v2` from
`baracuda-kernels-sys`. Keep the public `flash_attn(Tensor, ...)`
signature stable so the ~14 transformer files require zero edits.

**Pros**
- Smallest blast radius (single crate, no transformer churn).
- Preserves the existing BSHD axis ordering the eager wrapper assumes
  (vs. the lazy IR's BHSD).
- The `flash-attn` Cargo feature can stay (now means "compile the
  baracuda-backed eager wrapper").

**Cons**
- Keeps the `fuel-flash-attn-cuda` crate alive — one more crate to
  maintain.
- Two parallel FA2 launchers exist (the lazy one in
  `fuel-cuda-backend/src/flash_attn.rs` and the rewritten eager one) —
  the latter is essentially a copy of the former.
- The eager wrapper does its own BSHD validation that overlaps with
  baracuda's `can_implement_v2`; ~700 LOC of glue becomes redundant.

**Effort:** ~half-day to rewrite `cuda_fwd_t`. Validation gate: the
existing `fuel-flash-attn-cuda/tests/flash_attn_tests.rs` (live GPU)
plus selected transformer smoke tests.

### Option B: migrate transformers off the eager wrapper

Drop `fuel-flash-attn-cuda` entirely. The ~14 transformer files swap
their `fn flash_attn(...)` body from
`fuel_flash_attn_cuda::flash_attn(q, k, v, scale, causal)` to
`Ok(q.flash_attn(k, v, None, scale, causal, None, None, None))`
(the inherent method on `fuel_graph::Tensor` that emits
`Op::Fused(FLASH_ATTN, _)` and dispatches via the trait method through
`GraphExecutor`).

**Pros**
- One FA2 path total. Architecturally clean.
- Deletes `fuel-flash-attn-cuda{,-sys}` + the cudaforge build dep.
- All the BSHD-vs-BHSD validation collapses into `Op::FlashAttn`'s
  graph-level rank check.
- The `flash-attn` Cargo feature can be removed everywhere.

**Cons**
- Touches 14 transformer files, each behind a different model's
  feature gate. Per-model integration tests usually don't cover the
  flash-attn path; CI may not catch BSHD/BHSD axis-order bugs.
- The lazy IR's `Op::FlashAttn` takes BHSD (`[B, Hq, Sq, D]`); the
  eager wrapper takes BSHD (`[B, Sq, Hq, D]`). Every call site needs
  a `.transpose(1, 2)` insertion at minimum, possibly more depending
  on the surrounding tensor flow.
- Requires verifying each model's tensor flow before/after the
  attention call rather than trusting the wrapper.

**Effort:** ~1-2 days. Per-model verification can be parallelized but
requires care.

## Recommendation

**Option A first**, then revisit Option B once the lazy IR ships
fully. Reason: Option A is contained and unblocks crate-retirement; it
lets us measure whether anyone actually uses the `flash-attn` feature
in production (vs. the lazy path, which is the future). If consumer
data later shows the eager wrapper is unused, Option B becomes a one-
session cleanup. If consumer data shows heavy use, the rewritten
eager wrapper is the long-term home.

## Out of scope for this session prompt

- baracuda BW (`fa2_sdpa_*_run_bwd_v2`) integration. BW is only on the
  original 6 head_dims {32, 64, 96, 128, 192, 256} in baracuda
  alpha.60 per VENDOR.md, and Fuel-side `Op::FlashAttnBackward` does
  not exist as a graph variant yet. Wait for the in-place + autograd
  work to clarify backward shape before tackling.
- SD 1.5 hd40/hd80 path. Decided 2026-05-30: skip the baracuda
  escape-hatch ask, route those callers through standard attention.
  No action needed.
