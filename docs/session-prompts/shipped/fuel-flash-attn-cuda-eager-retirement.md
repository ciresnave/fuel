# Retire the eager `fuel-flash-attn-cuda{,-sys}` crates

> **SHIPPED 2026-06-20 (Option C).** Both crates deleted along with their
> workspace members, `workspace.dependencies` entries, the now-dead
> `cudaforge` workspace dep, and every `flash-attn` Cargo feature /
> optional dep across fuel-cuda-backend / fuel-core / fuel-transformers /
> fuel-examples / fuel-book. The baracuda FA2 launcher
> (`fuel-cuda-backend::flash_attn::launch`) was **preserved** — it was
> only feature-gated to reach the eager `-sys` crate; it now compiles
> unconditionally (still `#[allow(dead_code)]`, staged for the dispatch
> wiring it never had). The retired transformer models under
> `_models_retired/` keep their `#[cfg(feature = "flash-attn")]` arms,
> which are now permanently false (feature gone) so the
> `unimplemented!()` arm wins — harmless, they aren't compiled.
>
> Reconciled 2026-06-15 against the 2026-06-14 redirection + current git:
> still ACTIVE (the crates + `flash-attn` feature are not yet deleted), but
> Phase H (commit `cfcb35cf`) retired the ~14 transformer consumers under
> `fuel-transformers/src/_models_retired/`, so the path is now "delete the
> dead eager crates + feature outright" rather than migrate live callers.

## State entering this session

- **Lazy `Op::FlashAttn` path** — migrated to baracuda
  `fa2_sdpa_*_run_v2` in commit e359b041. Live oracle
  (`fuel-core/tests/flash_attn_cuda.rs`) passes F16 basic + causal on
  RTX 4070. baracuda alpha.60 vendors the full head_dim set
  {32, 64, 96, 128, 160, 192, 224, 256, 512} for FW.
- **Eager `fuel-flash-attn-cuda::flash_attn(Tensor, ...) -> Tensor`** —
  STILL calls `fuel_flash_attn_cuda_sys::run_mha` under the hood
  (986 LOC in `fuel-flash-attn-cuda/src/lib.rs`). It *was* used by ~14
  transformer model files behind their own `#[cfg(feature = "flash-attn")]`
  gates; as of Phase H (commit `cfcb35cf`) those files moved to
  `fuel-transformers/src/_models_retired/` and are no longer in the
  workspace build. The only remaining live callers of this API are the
  crate's own tests (`fuel-flash-attn-cuda/tests/flash_attn_tests.rs`).
- **`flash-attn` Cargo feature** — kept alive as transitional: now
  enables BOTH the migrated lazy launcher (no nvcc cost) AND the
  unmigrated eager wrapper (full nvcc build of vendored FA2 sources).

## Consumer inventory (the ~14 transformer files) — STALE as of Phase H

> **STALE (2026-06-15).** Every file listed below now lives under
> `fuel-transformers/src/_models_retired/` (commit `cfcb35cf`, "Phase H —
> eager fuel-transformers/models retired") and is invisible to the workspace
> build. None of them is an active consumer anymore. The list is retained
> only as a historical record of who *used* to call the eager wrapper; the
> only remaining live reference to the eager `flash_attn` API is the crate's
> own test suite, plus the lingering `flash-attn` Cargo features
> (`fuel-transformers/Cargo.toml`, and the gates in `fuel-core` /
> `fuel-cuda-backend`).

These all *previously* called
`fuel_flash_attn_cuda::flash_attn(q, k, v, scale, causal)`
behind `#[cfg(feature = "flash-attn")]` (paths shown at their pre-retirement
locations under `src/models/`):

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

### Option A: rewrite the eager wrapper to call baracuda directly — OBSOLETE rationale

> **Rationale obsolete as of Phase H (2026-06-15).** Option A existed to keep
> the eager wrapper alive *and* let us measure whether the `flash-attn`
> feature was used in production. With the ~14 transformer consumers retired
> (see the stale inventory above), there is no production usage signal left to
> measure, and no live caller worth preserving the wrapper for. Retained below
> as historical context only; see the Recommendation for the current path.

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

### Option B: migrate transformers off the eager wrapper — now largely MOOT

> **Mostly moot as of Phase H (2026-06-15).** This option was written when
> the ~14 transformer files were *active* consumers that needed a careful
> per-model migration to the lazy `Op::FlashAttn` path. They are now retired
> (under `_models_retired/`, not in the build), so there is nothing live to
> migrate. If/when a retired model is resurrected as a lazy port, it should
> emit `Op::FlashAttn` directly rather than route through the eager wrapper —
> but that work belongs to each model's lazy-port session, not to crate
> retirement. The retirement itself no longer depends on this migration.

For the historical record: dropping `fuel-flash-attn-cuda` entirely would
have meant the ~14 transformer files swap
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

**Option C: delete the dead eager crates and the `flash-attn` feature
outright.** Both Options A and B were framed for a world with *active*
eager consumers; Phase H removed them. With the ~14 transformer files
retired and the only remaining caller being the crate's own test suite,
there is no live wrapper left to preserve (Option A) and nothing live to
migrate (Option B).

In particular, the original Option-A rationale — "keep the wrapper around
so we can measure whether anyone actually uses the `flash-attn` feature in
production" — is now obsolete: the consumers that would have generated that
usage signal are out of the build, so there is no measurement left to take.
The lazy path (`Op::FlashAttn` → baracuda `fa2_sdpa_*_run_v2`) is already the
shipping FA2 launcher and stays untouched by this work.

Concrete steps for the deletion:

- Delete the `fuel-flash-attn-cuda` and `fuel-flash-attn-cuda-sys` crates
  (including `fuel-flash-attn-cuda/tests/flash_attn_tests.rs`, the last live
  caller) and their workspace members / path deps.
- Remove the `flash-attn` Cargo feature and the `dep:fuel-flash-attn-cuda`
  optional dependency from `fuel-transformers/Cargo.toml`, and drop the
  transitional `flash-attn` gates kept in `fuel-core` and
  `fuel-cuda-backend` (where they exist only to reach the eager wrapper — do
  NOT disturb the lazy launcher in `fuel-cuda-backend/src/flash_attn.rs`).
- Drop the now-dangling cudaforge / vendored-FA2 build dependency that only
  the eager `-sys` crate pulled in.

Validation gate: a clean `-p fuel-transformers` / `-p fuel-cuda-backend`
build with the feature gone, and the lazy oracle
`fuel-core/tests/flash_attn_cuda.rs` still passing on the RTX 4070.

## Out of scope for this session prompt

- baracuda BW (`fa2_sdpa_*_run_bwd_v2`) integration. BW is only on the
  original 6 head_dims {32, 64, 96, 128, 192, 256} in baracuda
  alpha.60 per VENDOR.md, and Fuel-side `Op::FlashAttnBackward` does
  not exist as a graph variant yet. Wait for the in-place + autograd
  work to clarify backward shape before tackling.
- SD 1.5 hd40/hd80 path. Decided 2026-05-30: skip the baracuda
  escape-hatch ask, route those callers through standard attention.
  No action needed.
