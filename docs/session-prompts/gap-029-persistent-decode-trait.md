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

### 3.3 The falsification attempt (GAP-166) — attempted, shown, and it SPLITS

Ruling (architect, 2026-08-12): design the vocabulary *against* DeepSeek2, do not
port it. The deliverable is an attempt to **express** MLA in the vocabulary, on
**both** axes, with the concrete text shown — because "I considered it and it
fits" is the same artifact as not having tried.

**Result: axis 1 passes with a caveat that "it fits" would have concealed;
axis 2 does NOT, and the failure is not where it looks.**

#### Axis 1 — state kind: EXPRESSIBLE, and the vocabulary already exists twice

I did not need to invent a describing vocabulary. One is shipped, in two
implementations:

- `LazyLatentCache::new(anchor, n_layers, max_seq_len, slot_trailing: Vec<Vec<usize>>, dtype)`
  (`lazy_latent_cache.rs:74`), with readers `n_slots()` (`:226`) and
  `slot_trailing(slot)` (`:228`).
- `LatentKvCache::with_capacity(n_layers, max_seq_len, slot_trailing: Vec<Vec<usize>>, dtype, device)`
  (`inference_context.rs:651`) — the device-resident twin, in the *same file* as
  `KvCache`.

Both state kinds expressed in it, literally:

```rust
// A standard KV layer — 2 slots, K and V:
slot_trailing = vec![vec![n_kv_heads, head_dim],
                     vec![n_kv_heads, head_dim]];

// An MLA layer — 2 slots, compressed latent and post-RoPE positional key:
slot_trailing = vec![vec![kv_lora_rank],
                     vec![qk_rope_head_dim]];
```

So **`KvCache` is the 2-slot special case of the slot vocabulary**, and the
`(n_kv_heads, head_dim)` pair is not a more primitive fact than
`slot_trailing` — it is one inhabitant of it. That is the strongest form of the
argument for the describing spelling: it is not speculative generality, it is
the generalization the tree *already made twice* and that `DecodeModel` did not
adopt.

**The caveat, which is the reason to actually attempt the expression rather than
assert the conclusion.** `slot_trailing` is **one list applied to every layer** —
`LazyLatentCache::new` loops `for _ in 0..n_layers { for trailing in &slot_trailing { … } }`.
So it describes **state KIND** generically while still **asserting uniformity
ACROSS LAYERS**. It solves the MLA axis and leaves the Gemma3/LFM2 axis exactly
where it was. Adopting it unchanged would buy one of the two dimensions and feel
like buying both.

**Therefore the vocabulary this trait needs is `slot_trailing` INDEXED BY
LAYER** — `fn layer_state_spec(&self, layer_idx: usize) -> LayerStateSpec`
rather than a single model-wide list. Note this lands on the architect's
independent constraint from the other direction: *`layer_idx` must be a real
parameter, not a loop counter*. Two separate lines of reasoning converge on the
same signature, which is the closest thing to corroboration available here.

#### Axis 2 — ownership shape: DOES NOT carry over, and the direction is inverted

This is the axis flagged as the one that would slip, and it did — my first
reading of it was backwards.

| | MLA (`forward_with_latent_cache`) | `DecodeModel` |
|---|---|---|
| threading | `cache: LazyLatentCache` **by value**, returned in the `Ok` tuple | `cache: &mut KvCache` |
| on `Err` | cache is **consumed and dropped** — caller loses it | caller **retains** the cache |

These are **not the same guarantee, and neither dominates**:

- By-value makes a partially-mutated cache *impossible* — but it also makes
  recovery impossible, because the handle is gone.
- `&mut` permits recovery, and is what the scheduler's documented contract
  actually relies on: the batched arm "may return an error (never a panic)
  **before mutating any cache**", so `advance_batched` can route that session to
  the serial arm **with its KV untouched**. Under by-value there is no cache left
  to route.

I assumed at first that by-value was the stronger guarantee and that adopting
`&mut` would silently drop it. **Measured, that is wrong in both halves.** All
six validation errors in `forward_with_latent_cache_impl`
(`lazy_deepseek2.rs:563`+) fire *before* any mutation — so MLA does satisfy
all-or-nothing, but via **early validation**, a property entirely independent of
the by-value threading. The by-value form is carrying something else: it is how a
*persistent/functional lazy structure* is rebuilt per step, the same reason
`advance_by(mut self) -> Self` (`:200`) is by-value. It is a data-structure
idiom, not an error-semantics choice.

**Consequence for the trait.** A trait requiring `&mut Self::State` *can* serve
MLA — `mem::replace` a placeholder out, call the by-value impl, write the result
back — but that adaptation **changes MLA's error contract** (the cache now
survives a failed step instead of being dropped) and must be written
deliberately, with the recovery semantics chosen rather than inherited. It is
expressible; it is not free; and it is invisible if you only check that the
*shapes* line up.

This is the [[replacing-a-lock-enumerate-incidental-guarantees]] pattern in a new
costume: two signatures that look like a mechanical `&mut`-vs-by-value
conversion are carrying different incidental properties, and the conversion is
sound only once you say which one you are keeping.

#### Verdict

**Do not add methods for DeepSeek2's benefit** (per the ruling, and nothing here
requires it). Take exactly two things from the attempt:

1. **Geometry is a per-layer slot spec, not a `(n_kv_heads, head_dim)` pair** —
   `LayerStateSpec` indexed by `layer_idx`. Uniform-KV models return the same
   spec for every layer, so all 8 in-scope families are unaffected in behaviour.
2. **State ownership is `&mut Self::State`**, chosen because the scheduler's
   error-recovery path needs the caller to retain the cache — and recorded here
   as a *choice with a named cost*, not as the obvious default.

Neither widens the increment. Both would have been invisible on a corpus of
eight uniform-KV models, which is the whole reason the check was run against
MLA: eight models that all satisfy both spellings **cannot distinguish them**,
and would have returned a clean `8 of 8` certifying nothing.

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

### Increment 2 — measured design, ready to build

**2a — the vocabulary. DONE (`0ef5ae76`).** `fuel-core/src/decode_state_spec.rs`:
`LayerStateSpec` / `StateSlot` / `collapse_uniform`. Sabotage-validated (see the
in-file record). 8 tests green.

**2b — the driver de-duplication. Designed, not built.** The two copies of
`rebind_and_realize_prebuilt` (`lazy.rs:9542` Llama, `:12114` Phi) are **48 lines
each and structurally identical**, differing in exactly three places:

| # | difference | resolution |
|---|---|---|
| 1 | Llama takes `rope_inv_freq: Option<&[f64]>`; Phi does not | per-model opts, carried by the hook |
| 2 | Llama passes `cache_dtype` + `s.offset_node().is_some()` to the token-data builder | absorbed into the hook |
| 3 | Llama calls `s.per_token_sym_env(cached_len)`; Phi binds `cached_len_sym` inline | **dissolves — see below** |

**Difference 3 dissolves, and that is a measured result rather than a
simplification.** `per_token_sym_env` (`inference_context.rs:1429`) is a method
on `DecodeSession`, a type **both** models already share; it binds
`cached_len_sym` *and* `attended_len_sym`. Phi's session already carries
`attended_len_sym = SymId(1)` (`lazy.rs:12088`), documented there as "carried for
API parity but never referenced/bound". Llama's own doc records that its
attended-length binding "is unreferenced on today's f32 decode graph (no flash
arm)" — so **Llama is already binding an unreferenced symbol harmlessly**, and
Phi doing the same is the identical no-op.

*Risk — RETIRED BY MEASUREMENT (`6a5eb8a4`), not argued away.* The claim was
sound only if `SymId(1)` is genuinely unused rather than merely undocumented, and
**the byte-exact tests cannot settle it**: a referenced symbol bound to its usual
value passes them, so the comment and the passing test are the same evidence.
The discriminating instrument is a deliberately **wrong** binding.

**The first attempt failed its own positive control on both models, and that was
the finding.** Perturbing `cached_len_sym` — the supposed known-referenced symbol
— did not move the output, so the test could not have detected referencedness at
all. Cause, measured: on the device-offset path the KV write offset rides a
device-resident **buffer** (`Op::WriteSliceDoff` reads the start from
`DecodeTokenData::offset` at launch), not the symbol. The proof-of-sight was
itself blind.

Rebuilt with control **A** (perturb the input token — must always move the
output) and control **B** (assert the direction matching the session's path
rather than assuming one). Result:

```
phi:   offset_node.is_some()=false   (SymEnv path;        SymEnv live=true)
llama: offset_node.is_some()=true    (device-offset path; SymEnv live=false)
```

**The two models take different paths, and Phi carries the evidence.** Phi runs
the SymEnv path where control B passed — a wrong `cached_len_sym` *does* move its
output — so Phi's SymEnv is demonstrably **live** while `attended_len_sym` is
**inert**. A sibling symbol in the same env is load-bearing and this one is not.
That is positive evidence that Phi adopting `per_token_sym_env` is a real no-op.

**Llama's arm proves strictly less.** Its whole SymEnv is inert on the
device-offset path, so it establishes only *"no symbol drives this path"* — same
consequence, weaker claim, **not citable about referencedness in either
direction**. Two greens are not two confirmations.

Scope, per the doc's own qualifier: **F32 / CPU / no flash arm.** A bf16/f16 CUDA
decode offering the flash arm would reference `attended_len` and must re-run this
control.

So the shared driver needs **one** hook, not two:

```rust
let data = model.build_token_data(&device, cached_len, tokens, s, cache)?;  // HOOK
let sym_env = s.per_token_sym_env(cached_len)?;                            // shared
let logits  = s.realize_token(&device, data, &sym_env)?;
```

**The oracle is confirmed present, which is what makes 2b safe to attempt.** The
rationale for porting the two existing impls first was that their tests are the
oracle; that is now measured, not assumed. Phi carries **8** decode tests
(`lazy.rs:18756, 18806, 18839, 18888, 18953, 19064, 19182, 19278`) and Llama the
parallel set (`:13652, 14462, 14858, 15351`). Two matter most:

- `phi_persistent_plan_once_matches_d1` (`:18953`) — the persistent/rebind path
  against the rebuild-every-token path, which is exactly what a botched
  extraction breaks.
- `phi_generate_loop_persistent_byte_exact_and_plans_once` (`:19064`) — asserts
  **byte-exact** output *and* plans-once, so it already walks the multi-token
  **rebind** path rather than only the build path.

*Residual to watch:* `cache.bump_version(li, KvSlot::K/V)` in the driver
hardcodes two KV slots. Correct for all 8 in-scope families and left alone
deliberately — generalizing it reaches into `KvCache`, i.e. the allocator, which
§3.3 puts out of scope.

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
