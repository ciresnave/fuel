# GAP-029 — generalizing persistent-KV decode off `LlamaModel`

**Status:** design approved (architect, 2026-08-08). Increment 1 landed;
increments 2+ not started.
**Scope ruling:** CireSnave scoped GAP-029 to **all 11** `lazy_quantized_*`
families, overriding a narrower architect ruling. This document covers the
**7 causal-LM families** that ruling resolved to; LFM2 and T5/Whisper are
tracked sub-scopes, not declines (see §6).

---

## 1. What the gap actually is

GAP-029 was filed as *"13 quantized model families implement neither decode
trait."* Two corrections, both measured:

- It is **11** families, not 13 (`ls fuel-core/src/lazy_quantized_*` → 11 files,
  11 `Quantized*Model` structs; a 12th `QuantizedFluxModel` exists in
  `lazy_flux.rs` but is a diffusion model, not a decoder).
- **Quantization is not the blocker for 9 of the 11.** Ten are thin newtype
  wrappers — `pub struct QuantizedXModel { inner: XModel }` — and the blocker is
  that the **unquantized inner model** has no persistent-KV decode either.

  Positive-controlled: `pub fn .*(cache|Cache|kv)` returns **0 hits** across
  `lazy_{gemma3,glm4,lfm2,phi3,qwen2,qwen3,qwen3_moe,smollm3,t5}.rs` and **2
  hits** in `lazy_llama_full.rs` (the known-present target), so the search can
  find what it is looking for.

Across all of `fuel-core`, `forward_with_kv_context_persistent` exists on exactly
**three** types: `LlamaModel` (`lazy.rs:8885`), `PhiModel` (`lazy.rs:11859`),
`Llama3Model` (`lazy_llama_full.rs:382`).

> **Correction (2026-08-12).** This line previously cited `lazy.rs:11420` for
> `PhiModel`. That anchor is wrong — `:11420` is inside `impl PhiConfig`'s
> `from_json` field parsing. `impl PhiModel` opens at `:11512` and the persistent
> fn is at `:11859`. The *claim* was right and the *pointer* was not, which is the
> more dangerous combination: it survives review because the sentence is true.
> Re-verified against `origin/main` after 73 commits, **zero** of which touched
> these files (`git log <base>..origin/main -- <files>` → empty), so the drift was
> mine at authoring time, not the tree's since.

**So this is a decode-coverage gap that would exist identically if quantization
were deleted.** A reader acting on the original row would have gone looking in
GGUF internals for a problem that lives in the base models.

## 2. The seam (measured, not estimated)

`build_and_realize_first_decode_token` touches only:

| kind | members |
|---|---|
| scalar dims | `n_layers`, `dim`, `vocab_size`, `head_dim`, `n_kv_heads`, `norm_eps` |
| weights | `token_embedding`, `output`, `layers`, `final_norm_gain` |
| methods | `decode_shape_key`, `build_token_rope_mask_arcs`, `apply_layer_with_kv_writes` |

**Those three methods are the variation points, and that is evidence rather than
judgement:** `PhiModel` already carries its own copy of
`rebind_and_realize_prebuilt` (`lazy.rs:12114`), `decode_shape_key` (`:11514`),
and both hooks (`:11561`, `:12189`) parallel to `LlamaModel`'s.

**There are THREE carriers on `main` today, not two — and the third diverged
rather than duplicated.** `lazy_deepseek2.rs` carries `decode_shape_key`
(`:499`) and `rebind_and_realize_prebuilt_mla` (`:1373`), but **no**
`apply_layer_with_kv_writes`; its decode entry point is
`forward_with_latent_cache` (`:563`). So the machinery has already been copied
twice and *forked* once. That is worse than three identical copies: a diverged
carrier cannot be collapsed by deletion, only by design.

Eight more ports would make it ten. Which is why this is **one refactor with
eight consumers**, not eight ports — and why the refactor has to happen before
the ports, not after.

## 3. The constraint that decides the trait's shape

```rust
fn apply_layer_with_kv_writes(
    &self, x, layer: &LayerWeights, k_cache_const, v_cache_const,
    cached_len_sym, attended_len_sym, offset,
    rope_cos, rope_sin, mask,      // <-- ONE set, per MODEL
) -> Result<LazyTensor>
```

`rope_cos` / `rope_sin` / `mask` are per-**model**. **Gemma3 violates exactly
this**: layer `i` alternates a sliding-window mask + the *local* RoPE base
against full-causal + the *global* base
(`lazy_gemma3.rs:110-112`, `:61-66`).

Two consequences:

1. **`layer_idx` must be threaded into the layer hook**, and `DecodeTokenData`
   must carry **N rope/mask variants** (N=1 for LLaMA, N=2 for Gemma3) instead
   of one. Cheap now; retrofitting later touches every impl.
2. **Qwen3Moe needs nothing extra** — MoE routing lives inside the per-layer
   FFN, and the existing hook already has that granularity. It was on the
   "hard" list because MoE *sounds* structural; it isn't.

### 3.1 Describe, do not assert — the binding constraint

**A geometry method must describe _the state this layer requires_, never assert
_the KV shape for every layer_.** The two spellings have the same method count
and radically different futures:

- *"the KV shape for every layer"* → LFM2 changes the **meaning** of the method
  for all impls. Rewrite.
- *"the state this layer requires"* → LFM2 adds a **variant**. Extension.

Adopt the second **even where it looks like over-generality today**. Gemma3 is
live evidence that per-layer variation is real rather than speculative:
Gemma3 varies per-layer *behaviour*, LFM2 varies per-layer *state kind* — the
**same axis**, different payload. The expensive part (making `layer_idx` a real
parameter rather than a loop counter) is paid once, here.

### 3.2 The constraint is already violated on `main` — MLA is the proof

`DecodeModel` (`fuel-inference/src/multi_session.rs:77`) requires
`n_kv_heads()` and `head_dim()`, both documented "(cache geometry)".
**`DeepSeek2Model` is decode-capable on `main` today and has neither**, because
MLA's decode state is a `LazyLatentCache`, not a `KvCache`: per layer, slot 0 is
the post-norm compressed latent trailing `[kv_lora_rank]`, slot 1 the post-RoPE
`k_pe`. The signature diverges too —
`forward_with_latent_cache(&self, tokens, cache: LazyLatentCache) -> Result<(LazyTensor, LazyLatentCache)>`
threads the cache **by value and returns it**, where the trait takes
`&mut KvCache`. Different state kind *and* different ownership shape.

**The hazard is not that DeepSeek2 cannot implement the trait. It is that it
can, and wrongly.** `DeepSeek2Config` has `num_attention_heads`, and a
`v_head_dim` exists, so both methods are syntactically returnable. A scheduler
trusting them allocates a standard `KvCache` of `[n_kv_heads, head_dim] ×
n_layers` for a model whose decode path never reads a `KvCache` at all. It
type-checks, it runs, it allocates the wrong state — the exact LFM2 failure mode
(§6), except **shipped rather than prospective**.

Note where the conflation lives, because it is one sentence. `n_layers`' doc
reads *"the KV cache geometry the scheduler builds a session's private cache
against (all sessions share one model, so this is uniform)."* That parenthetical
is a **correct justification for uniformity across SESSIONS**; the methods use it
to assert uniformity **across LAYERS and across STATE KINDS**, which it does not
establish. A true reason carrying a wider claim than it supports.

**Design consequence for increment 2.** MLA is the only *shipped, working,
non-KV* decode path in the tree, so it is the one consumer that can falsify
"describe, do not assert" **before** eight impls exist — LFM2 cannot, since
nothing is built there. It is **out of GAP-029's scope** (no
`lazy_quantized_deepseek2.rs`; positive-controlled against the 11-file listing)
and is **not** to be ported here. It is a *design check*: can the geometry
vocabulary express `LazyLatentCache` without lying? Failing that check costs two
impls now and ten later.

*(Fact vs consequence: the file:line anchors, cache type and signature are
measured off `origin/main`. That a scheduler **would** mis-allocate is reasoning
from the trait's stated contract, not an observed run.)*

## 4. Increment order

**Increment 1 — DONE (`8e915e19`).** `DecodeModel for QuantizedLlama3Model`,
delegating to `self.inner()` (the scaling-aware `Llama3Model`), never to
`self.inner().inner`. Verified: `|A-B| = 0.000e0`, `|B-C| = 2.118e-3`.

**Increment 2 — the de-duplication, and it is the right first refactor.**
Define the trait and port the **two impls that already exist** (`LlamaModel`,
`PhiModel`). Do this *before* adding any new family, because both already work
and their existing tests are the oracle: the refactor is correct exactly when
their behaviour is unchanged. **A new family cannot serve as that oracle — it
has no prior behaviour to be unchanged from.** Net effect is deleting a copy,
not adding abstraction on spec.

**Increment 3 —** the 6 LLaMA-shaped families: Qwen2, Qwen3, Qwen3Moe,
SmolLm3, Glm4, Phi3. Each = `apply_layer` for its architecture + the quantized
wrapper's delegation.

**Increment 4 —** Gemma3, which is what forces the N-variant rope/mask and
`layer_idx` machinery to actually be exercised rather than merely present.

## 5. Gates

- **`--all-targets` per constructing crate**, not `--lib`. A trait impl can
  break test targets `--lib` never compiles (paid for once already).
- **Logits, not sampled tokens.** This module's tiny model has a fixed-point
  greedy argmax, and increment 1 measured the wrong-arm separation at
  **2.118e-3** — right at the scale seeded temperature sampling is documented to
  swallow. A token oracle would have seen nothing.
- **Every parity test needs a negative control**, asserted *first*, that the
  correct and incorrect arms are distinguishable at the tested position. A
  parity assert without one passes whichever arm is wired.
- **Decode more than one token.** The first decode token BUILDS the held graph;
  later tokens REBIND it. A 1-token test is blind to a model that is correct at
  token 1 and wrong from token 2 — the documented hazard in
  `forward_with_kv_context_persistent_inv_freq`.

## 6. Out of scope here — tracked, not declined

- **LFM2 → GAP-098.** It interleaves GQA attention with **ShortConv (LIV)**
  blocks (`lazy_lfm2.rs:4, 17-33, 61-63`) whose decode state is a rolling
  window of `conv_kernel_size` inputs, not KV. `KvCache::with_capacity`
  allocates one uniform geometry × `n_layers`, so **allocating KV for a
  ShortConv layer is the wrong state, not wasted memory** — a conv ring buffer
  and a KV block are different objects with different lifetimes, and a contract
  that lets you allocate the wrong one is a contract that will be misused.
  Architecture §14 question; needs its own design.
- **T5 / Whisper → own row.** Cross-attention KV has **different lifetime
  semantics**: self-attn KV is *appended every decode step*, cross-attn KV is
  *computed once from the encoder and read-only for the whole generation*. A
  trait method meaning "extend the cache with this step's keys" **has no
  meaning** for cross-attn — not a parameter difference, a different contract.

## 7. Related

GAP-007 (Q4_0/Q4_K_M CUDA `M=1` guard) is **not** a blocker for this work.
Measured on the 4070 (`e8d63a2a`): dispatch does not route a Q4_0 `m>1` matmul
into that guard, so quantized prefill does **not** hard-fail on CUDA
(`scale_rel` 1.97e-4 at m=1 vs 4.79e-4 at m=6 — same accuracy class). The
earlier caveat that this work would land "CPU/Vulkan-correct and
CUDA-prefill-broken" is **false**.
