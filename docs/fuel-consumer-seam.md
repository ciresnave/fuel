# The Fuel consumer seam — annexes, audit, and open questions (2026-07-28)

**Status.** Phase doc. **The core of this document was promoted to the constitution on
2026-07-28** as [15-consumer-contract](architecture/15-consumer-contract.md) — the rule, the
fungibility derivation, the two directions, clauses C-1…C-7, the refusals, and the consumer-class
matrix now live there and are authoritative. Per
[`architecture/00-index.md`](architecture/00-index.md#how-phase-docs-relate-to-this-set), this doc
cites rather than restates.

**What stays here**: the per-consumer-class annexes, the as-built audit of the shipped serving
substrate, the folds-into table recording which use cases *don't* earn a class, and the open
questions. These are working detail, not steady-state architecture.

**Origin.** Lightbulb (an inference engine currently on Candle) will be ported onto Fuel, and Fuel
had meanwhile shipped Increment 1 of a serving substrate
([`fuel-core/src/multi_session.rs`](../fuel-core/src/multi_session.rs)). That made the boundary
question concrete: without a stated rule, a scheduler making fairness decisions lands inside Fuel,
and the ported consumer arrives with a second scheduler that loses to it.

---

## The rule, in one line

**Fuel owns mechanism; the consumer owns policy — Fuel must be preemptible, accountable, and
manageable, and must never decide whose work matters.** The discriminator, the derivation from cost
fungibility, and the seven clauses are in
[15-consumer-contract](architecture/15-consumer-contract.md). Everything below assumes them.

---

## What folds in, and where

Recorded so the filtering is auditable rather than re-litigated. None of these earn a class; each is
an instance of one in [15's class matrix](architecture/15-consumer-contract.md#consumer-classes).

| Use case | Folds into | Wrinkle worth remembering |
| --- | --- | --- |
| Fine-tuning / PEFT / LoRA | B (training), A (adapter serving) | shared immutable base + small mutable per-tenant delta; C-1 must account for adapter memory separately, C-3 evicts the adapter not the base |
| Hyperparameter search / experiment orchestration | *N* × B plus an orchestrator | the orchestrator is above the consumer layer, not a consumer; it is where cost stops being fungible |
| Speculative decoding / draft models | A | two models, **one** consumer — the arbitration unit is the VRAM pool; accept/reject is consumer policy |
| Distributed / multi-node training | B | collectives are Fuel's over a consumer-supplied topology; failure and elasticity response are the consumer's |
| Benchmarking / profiling | D | C-4-heavy; an oracle runner that cares about the measurement rather than the value |
| GPU array / preprocessing (Fuel as a tensor library) | C | wants the tensor surface without autograd — already a named crate-fission driver in [02-layers](architecture/02-layers.md) |
| Differentiable scientific computing / simulation | B-shaped | exact-restorable and iterative, but with no model and no "training"; a good check that B's clauses aren't secretly ML-specific |
| Model surgery / merging / pruning / export | **none — graph consumer** | governed by [03-ir](architecture/03-ir.md) + [13-interchange](architecture/13-interchange.md); see [15 §Scope](architecture/15-consumer-contract.md#scope-this-contract-governs-execution-consumers) |

---

## Annex A — Inference host (worked example: Lightbulb)

**Unit** a decode step over a ready set. **State** per-session KV, lossy-restorable (recomputable
from tokens). **Cost** tokens, KV bytes, GPU-seconds.

**Clause specialization.** C-1: free KV blocks, max sessions at a geometry, batch admissibility via
the uniformity gate. C-2: quantum = *n* tokens or a deadline; cancel at the decode-step barrier.
C-3: evict KV to host, or discard and mark the session recomputable — lossy is acceptable and
cheapness dominates. C-4: tokens produced, KV bytes resident, elapsed, arm used. C-5: **already live
and load-bearing** — the batched arm is documented as *ε-close* (logits within 1e-4) and
token-identical, so a consumer returning logprobs to users has a materially different requirement
from one returning only tokens, and must be able to demand the bit-exact serial arm.

### A.1 As-built audit — `fuel-core/src/multi_session.rs`

Increment 1 shipped a good substrate; this is about placement, not quality. **[verified]** = read
from the code, **[judgment]** = assessment.

**Correctly Fuel's:**

- **`SchedulePolicy::{RoundRobin, Batched{max_batch}}` [verified].** Worth stating plainly, because
  the name invites the opposite reading: the two arms are *semantically equivalent* — the doc
  comment calls the batched arm "provably equal to `RoundRobin`", with `RoundRobin` as the byte-exact
  oracle. This is **not** scheduling in the fairness sense; it is selection among equivalent
  implementations, exactly [`frontier-paradigms-vision.md`](frontier-paradigms-vision.md)'s framing
  of `Op::Branch` ("plan-time selection among implementations of the same math… **not**
  data-dependent dispatch"). It stays. **[judgment]** the name should change (`DecodeArm`?) so
  fairness logic cannot grow into a slot that sounds like it invites it.
- **`SessionState`, `ModelDims`, `BatchOutcome`, the uniformity gate, per-session error isolation
  [verified].** Mechanism, correctly placed. Error isolation — a per-session `Err` finishes that
  session rather than killing the batch — is precisely the isolation property a consumer needs.

**Provisional — fine as an oracle, must not become the interface:**

- **`run_to_completion()` [verified]** drives every session to completion with no preemption,
  fairness, or yielding. Correct as a test/oracle driver; it is the exact shape a consumer must own.
  Keep it, mark it a harness convenience, and do not let a consumer call it.
- **`add_session()` [verified]** constructs and always accepts — it is *construction*, not admission.
  The name is the risk: admission logic will accrete there unless C-1 lands and it is renamed to
  reflect that it is unconditional.
- **Implicit FIFO [verified]** — `step()` advances sessions in `Vec` order. A fairness policy chosen
  by omission: invisible, unstated, and unoverridable. Order should be consumer-supplied.

**Clause status:** C-1 absent (no headroom query; OOM surfaces as an `add_session` error). C-2 absent
(no quantum, deadline, or cancel). C-3 absent — `KvCache` has no evict/restore path; **the
load-bearing gap.** C-4 partial — `StepReport` carries *what happened*
(`advanced`/`finished`/`errored`/`used_batched_arm`) but not *what it cost*. C-5 absent as a
consumer-facing control, though the underlying arm distinction exists.

### A.2 Layer drift — a separate, verified defect

**[verified]** `multi_session.rs` lives in `fuel-core` and takes `model: &'m LlamaModel` (~line 348)
plus `SamplingStrategy`. [`ROADMAP.md`](../ROADMAP.md)'s layer table states Foundation (`fuel-core`)
*"will never contain: tokenization, model-family assumptions, **serving abstractions**, HF Hub client
code"* — session lifecycle, sampling, and a Llama-specific model reference hit three of those
categories.

Per the working agreement ("treat doc-vs-code drift as a defect"), this should move up a layer
(`fuel-inference`, whose exclusion list permits it) or the layer table should be amended with an
argued exception. **Recommendation: move it**, before the Lightbulb port — the move is what forces
`&LlamaModel` to become a model-agnostic trait, which any inference consumer needs anyway since none
of them will serve only Llama.

### A.3 Lightbulb port survey (2026-07-29)

**Method.** Read-only structural survey of the *pre-port* codebase at `C:\Projects\Lightbulb`
(CireSnave's note: "it has not yet even started to be ported to Fuel. It will change a lot").
Structural, not behavioural — file/symbol counts and module shape, no execution. **[verified]** =
counted or read from code; **[judgment]** = assessment.

> **Standing caveat (added 2026-07-29, the hard way).** **Everything in A.3 and A.4 is an
> *existence* claim, not a *behaviour* claim.** `[verified]` here means "this symbol/file/count is
> really there," never "this runs." That distinction was stated in the Method line from the start
> and then violated anyway — gap #2 below was written as "working end-to-end binaries" on the
> strength of an `ls`, and the binary does not execute. By the port session's own count, **three
> structural findings in this survey did not survive contact with execution**: the `fuel-inference`
> "near-1:1 overlap" (the first three modules examined turned out complementary, not duplicative),
> a delete-list entry that was half mechanism and half policy, and gap #2. All three were
> well-evidenced *structurally* — file counts, symbol greps, module names.
>
> **Re-check the reverse gap list by execution before planning around it.** Structure tells you
> what to look at; it does not tell you what works.
>
> **And the same caveat applies to the *consumer* side of this survey (2026-07-29).** The port
> session reports that Lightbulb had **never compiled** before that day (missing workspace root,
> plus an `mlmf` dependency that could not have type-checked against `candlelight`), and that its
> correctness suite **still does not** — `batched_transformer_correctness` fails with 36 errors of
> genuine API drift. So the parity oracle the port plan depends on is not merely unbuilt but stale
> by an API generation. **Every claim in A.3/A.4 about Lightbulb's *behaviour* — as distinct from
> its structure — is therefore unverified**, including the H2O semantics that the reduction-in-graph
> spec is derived from. The structural claims (file counts, module shape, symbol presence) stand;
> anything about what the code *does* is pending that suite being restored.
>
> **A second failure mode, found the same day and not covered by the rule above.** The port
> session's first execution had a broken harness, and *both* the control and the test failed
> identically; had only the test been run, the result would have been a confident report of a Fuel
> gap that does not exist. The evidence-distance heuristic guards against overclaiming *from
> structure*; it does nothing against **concluding absence from a broken harness**. Only a control
> catches that. **Both disciplines are required, and they protect against opposite errors.**
>
> **The unifying form, after nine instances across two sessions:** *"I ran a command that answered a
> narrower question than the one I reported on."* It has three shapes, each needing a different
> guard:
>
> | Shape | Example | Guard |
> | --- | --- | --- |
> | **Existence read as behaviour** | `clear_slot` exists and is called → "slot reuse happens"; `ctor`/`inventory` absent → "no auto-registration" | run it |
> | **Truncation read as total** | a `head -10` grep → "nine sites" (25); "no test matches" (two do) | `wc -l` before trusting a `head` |
> | **Memory read as record** | "I read the constructor docs" → the transcript shows `grep -n` returning line numbers only | check the transcript, not the recollection |
>
> The third is the hardest, because the subject is the only witness and has no reason to doubt
> themselves. It surfaced here only because the consumer went back to their own transcript
> voluntarily — and it invalidated a priority ranking two parties had already acted on.

**Shape [verified].** ~66k LOC across 168 `.rs` files repo-wide, 110 under `src/`; single crate.
Built on **`candlelight`**, a Candle *fork* — not stock Candle. 44 files repo-wide touch the tensor
layer. **[flag]** what candlelight diverged from Candle is unknown to this survey and is a real port
input.

**The tensor coupling is concentrated in `model/` [verified].**

| Subsystem | files touching tensors | LOC | Character |
| --- | :-: | ---: | --- |
| `model/` | **16 / 17** | 8,895 | the tensor core — the actual port surface |
| `multi_gpu/` | 4 / 6 | — | placement / sharding |
| `cache/` | 5 / 12 | 9,399 | 5 tensor files; the other 7 are **policy** |
| `loaders/` | 3 / 3 | — | weight loading |
| `memory/` | 3 / 4 | 774 | |
| `engine/` | **1 / 25** | 15,954 | reasoning layer — reaches tensors through `speculative.rs` alone |
| `api/` | **0 / 9** | 2,288 | OpenAI-compatible server |
| `contracts/` | **0 / 6** | 1,331 | constrained generation |

**Finding 1 — Lightbulb is mostly not a tensor program [verified].** Its largest subsystem
(`engine/`, 16k LOC, 25 files) is *reasoning orchestration* — `query_analysis`, `knowledge_base`,
`context_injection`, `decomposition`, `relevance_search`, `conversation_history`, `tool_call` — and
touches the tensor layer through exactly one file (`speculative.rs`, i.e. speculative decoding, which
genuinely needs tensors). `api/` and `contracts/` are tensor-free entirely. Notably `model_runner.rs`
does *not* touch candlelight directly — the reasoning layer reaches models through `model/`'s
abstraction.

**Finding 2 — Lightbulb's shape independently corroborates the mechanism/policy line
[verified structure, judgment on significance].** It already partitions the way
[15](architecture/15-consumer-contract.md) predicts, without having been designed against it:

- `cache/{h2o_policy, streaming_policy, segmented_eviction_policy, eviction_policy}.rs` — eviction
  **policy**, four files, zero tensor coupling. Exactly what 15 says stays with the consumer.
- `cache/tiered_storage.rs` — C-3 externalization, consumer side, already built.
- `cache/{prefix_cache, cache_span}.rs` — shared-prefix reuse, i.e. the refcounted COW splice
  Increment 2 is building, from the consumer's side.
- `engine/{slot_pool, slot_monitor, memory_aware_scheduler}.rs` — the consumer's own admission and
  scheduling.
- `sampling.rs` + `contracts/` — sampling policy.

**[judgment]** The contract was derived from Fuel's side and Lightbulb sits on the predicted side of
every line. That is meaningful corroboration that the boundary is real rather than invented.

**Q5 RESOLVED — sampling is consumer policy [verified].** `src/sampling.rs` is 123 lines of
host-side post-processing over **realized** `&mut [f32]` logits: temperature scaling, top-k, top-p,
and a seeded `StdRng` draw. `contracts/` (`enum_choice`, `tagged_fields`, `validation`,
`commit_block`) layers constrained generation on top — sampling policy Fuel could never anticipate.
**Fuel should produce logits and stop.** `SessionState::sample_and_append` duplicates something the
consumer already owns and is richer at. It is scheduler-surface, so it rides the `fuel-inference`
move rather than Increment 2.

**Sub-finding — this bounds C-3-exact [verified].** Lightbulb re-seeds
`StdRng::seed_from_u64(seed)` *per sample call*; the consumer owns its RNG entirely. So the "RNG
stream position" an exact-fidelity C-3 handle must cover is **Fuel's** RNG (training-side sampling,
dropout), never the consumer's sampler. Worth pinning so the Exact arm doesn't over-scope.

**Q6 RESOLVED — the clauses do NOT gate the port.** Lightbulb keeps its own `engine/`, cache
policies, `memory/`, slot pool, scheduler, sampler, and API. The port is a **tensor-layer swap
concentrated in `model/`**, not a re-architecture around the consumer contract. C-1…C-7 are adopted
incrementally as Fuel offers them; **none block the port**.

**What does gate it — the reverse gap list (what Lightbulb needs *from Fuel*):**

1. **Eager → lazy.** ~~70 value-extraction sites, the single largest port risk.~~ **AUDITED
   2026-07-29 — the raw count was misleading and the real risk is elsewhere.** The port session
   classified all 73 sites:
   - **49 are inside `#[cfg(test)]`** — test assertions, not production, not on the decode path.
   - **~5 are dead debug code** — vestigial realizes whose consumers were deleted. The worst,
     `mlp_wrapper.rs:156` inside `MlpWrapper::forward()`, realizes the full activation tensor **on
     every MLP call, every layer, every token** to compute two statistics that are then discarded
     (`let _input_max = …; let _input_mean = …;`). Under Candle that is a wasteful copy; under Fuel
     it would break the graph at every MLP in every layer — fusion gone, capture impossible. Same
     pattern at `custom_transformer.rs:264`/`:550` and `custom_attention.rs:675`. **These get
     deleted, not ported.**
   - **~9 are legitimate realize boundaries** — `logits.argmax(0).to_scalar::<u32>()` → next token
     (the canonical one), tensor serialization, offline pruning/calibration.
   - **~24 is the real production surface; ~4 are a genuine design problem** — and all four reduce
     to a single issue, promoted to its own item below.

   So eager→lazy is **mechanical for nearly all of it**. The residue is not a translation problem;
   it is a clause problem (item 6).
2. ~~**Model implementations** — `fuel-transformers` parity needed.~~ **CORRECTED 2026-07-29 —
   mostly CLOSED, not a Fuel-side blocker.** Raised by the Lightbulb port session and **[verified]**
   here: `fuel-transformers/src/models/` does not exist; the eager ports are dead in
   `_models_retired/`. The live surface is `fuel-core`'s lazy layer — 157 `pub mod lazy_*` modules,
   with `fuel-core/src/lazy_llama_full.rs` (731 LOC) carrying `LlamaModel` as the canonical
   lazy-graph LLaMA decoder plus `Llama3Model` (line 319) with three-band Llama-3.1 long-context RoPE
   and `from_hf_json_str` for full HF `config.json`. Binaries exist in
   `fuel-lazy-examples/src/bin/`: `llama-lazy`, `llama-lazy-cuda`, `llama-lazy-vulkan`, `gemma-lazy`,
   `phi-lazy-vulkan`, `bert-lazy`, `convnext-lazy`. (The consumer's own
   `custom_transformer`/`custom_attention`/`custom_transformer_block`, ~3.3k LOC, still ports as
   ordinary graph code.)

   > **PARTIALLY RETRACTED 2026-07-29 — the binaries do not run.** This entry originally said
   > "**working** end-to-end binaries" and "the `model/` rewrite has a working reference to copy."
   > **Both claims were unearned**: existence was verified with a directory listing; execution
   > never was. The port session ran `llama-lazy` and it fails at the first realize with
   > `no backend supports matmul on [F32, BF16, F32]; available backends: []` — an *empty
   > registry*, not a dtype gap. It builds, downloads TinyLlama, loads weights, parses config, and
   > builds the graph, then dies at realize. Reproduced on current `main`.
   >
   > **~~Mechanism [verified here]~~ — RETRACTED 2026-07-29. My diagnosis was wrong on every
   > clause, and I gave it to the port session with confidence.** I wrote that registration is
   > "explicit", that there is "no `ctor`/`inventory`/`linkme` auto-registration", that linking
   > `fuel-cpu-backend` "does not register it", and that "the registry is empty because nothing
   > populates it."
   >
   > **[re-verified] `dispatch.rs:6554` — `global_bindings()` auto-registers**:
   > `GLOBAL_BINDINGS.get_or_init(|| { … register_cpu_kernels(&mut t);
   > register_optional_backends(&mut t); … })`, with the doc comment stating outright that "CPU
   > dispatch wrappers are auto-registered on first access… production callers picking up the
   > global table see all available backends **without manual init**." The registry is **not**
   > empty. I greped for three auto-registration *crates*, found none, and concluded there was no
   > auto-registration — when the mechanism is an ordinary `OnceLock` in the accessor, which needs
   > none of them.
   >
   > **Actual root cause (found by the serving/dispatch session, not by me):** an **unbuilt CPU
   > mixed-precision matmul kernel** — CPU registers only uniform `[T, T, T]`; the mixed
   > `[F32, BF16, F32]` form is CUDA-only. The `available backends: []` text is a **misleading
   > stub that ignores the real binding table**, which is what sent both of us down the
   > empty-registry path. (I could not locate that error string in `fuel-dispatch`, `fuel-graph`,
   > or `fuel-core`; I am not claiming to have found the stub — recording the attribution
   > accurately.)
   >
   > **Credit, stated precisely:** running the binary established **that** it was broken, not
   > **why**. The port session's empty-registry hypothesis was wrong; mine was wrong in the same
   > direction and with more confidence. This is the **fifth** structural finding in this survey
   > to fail on contact with execution, and the first one that is entirely mine — recorded because
   > the standing caveat above applies to the author of the caveat too.
   >
   > **~~A second reason not to lean on it~~ — WITHDRAWN 2026-07-29, that claim was wrong.** I
   > wrote that `llama-lazy.rs` "does not exercise the model surface this gap points a consumer
   > at" because it uses `lazy_llama2c::Llama2cModel`. The port session pushed back with file:line
   > evidence and is correct; **[re-verified here]** `lazy_llama2c.rs:10` — *"Thin wrapper over
   > [`crate::lazy::LlamaModel`]"*; `:66` — *"the forward delegates to [`LlamaModel`]"*; and every
   > forward (`:74`, `:86`, `:106`) constructs `LlamaModel { config: to_llama_config(), … }` and
   > delegates. It runs **real TinyLlama weights** (`dim=2048 layers=22 heads=32 kv_heads=4`)
   > through the llama2.c *config adapter*, not the llama2.c *architecture*. So it does exercise
   > `lazy::LlamaModel`, transitively and by design. **One axis wrong, not two.**
   >
   > **What is narrowly true, and matters more than the retraction:** that path is
   > `LlamaModel::forward(tokens, start_pos)` with **no KV cache** (`llama-lazy.rs:29` says so
   > outright), and it touches neither `Llama3Model` / `lazy_llama_full.rs`'s three-band RoPE nor
   > `InferenceContext` / `KvCache` / `CapturedRun`. So the honest split is:
   >
   > - the **decoder** surface has a runnable path (a valid smoke test), pending the dispatch fix;
   > - the **serving** path — KV cache, batched decode, capture-shaped replay — has **no runnable
   >   example at all**.
   >
   > **That second half is a genuine gap this entry would otherwise have recorded as closed**, and
   > it is the half an inference host actually needs. Raised by the port session; recorded here as
   > its own item rather than buried in a retraction.
   >
   > **CLOSED BY EXECUTION 2026-07-29 — tokens out of Fuel.** `llama-lazy` built clean at
   > `origin/main` `13279179` and generated on TinyLlama-1.1B: *"Once upon a time, in a land far far
   > away," — 8 tokens in 28.37s (3.55 s/tok)*, exit 0, coherent English. Same invocation that died
   > at the first realize that morning.
   >
   > **Scope, stated precisely — "Fuel runs a model" is broader than what was established:**
   > - **Verified by execution:** the *decoder* path end to end — weight load, config parse,
   >   tokenization, graph construction, realize, sampling, detokenization — on a real 22-layer
   >   model, through `lazy::LlamaModel` via the `Llama2cModel` thin wrapper. That is the surface a
   >   consumer's `model/` rewrite copies from, so it is now a **working reference**, as this entry
   >   originally (and unearnedly) claimed.
   > - **Still unverified:** the *serving* path. No KV cache in that binary, no batched decode, no
   >   `CapturedRun` replay, no `Llama3Model` RoPE scaling. **The serving path still has no runnable
   >   reference, and the first consumer writes it.** That half is unchanged and is the half an
   >   inference host needs.
   >
   > **[note] This is the first entry in the survey that neither party could have got wrong by
   > reading.** Reaching it took a build that exposed a misleading diagnostic, a wrong hypothesis
   > from the consumer, a correct root cause from the dispatch session, a capability decision from
   > CireSnave, a general optimizer fix rather than a per-consumer kernel, and then running the
   > thing. Nine existence-read-as-behaviour errors across two sessions in one day; the fix for
   > every one was the same.
   >
   > **Corrected status:** the lazy model surface **exists and appears complete** — 157 `lazy_*`
   > modules, `LlamaModel`, `Llama3Model` with three-band Llama-3.1 RoPE and an HF-config
   > deserializer are all really there, and the eager-retired-vs-lazy-live story holds. But the
   > **end-to-end example path is unverified-to-broken pending a dispatch fix**, and a consumer
   > cannot presently confirm parity by execution. Filed with the serving/dispatch session; fix
   > requested by CireSnave. Close this gap on a token coming out of Fuel, not on a file listing.
3. ~~**`nn` surface** — needs a coverage check.~~ **CLOSED 2026-07-29 (parity check, all
   [verified]).** Every symbol the consumer uses has a Fuel counterpart. `VarBuilder`/`VarMap` →
   `fuel_core::lazy_nn_varbuilder::LazyVarBuilder` / `lazy_nn_varmap::LazyVarMap`. `Linear` /
   `linear_no_bias` → `fuel-core/src/lazy_nn/linear.rs` (`linear()` :145, `linear_no_bias()` :169).
   `embedding` → `lazy_nn/embedding.rs`. `ops::silu` → `fuel-graph/src/lib.rs:4761` +
   `fuel-core/src/lazy.rs:498` (plus `silu_inplace` :4808). `ops::softmax_last_dim` →
   `fuel-graph/src/lib.rs:5970` + `fuel-core/src/lazy.rs:1037`. **Bonus overlap:** `lazy_nn/` also
   ships `lora.rs` (`LazyLoraLinear`) and `quantizable_linear.rs` (`LazyQuantizableLinear`), which
   duplicate the consumer's `lora/` and `model/quantizable_linear.rs` — add both to the A.4 diff.
   Full `lazy_nn/` inventory: activation, conv, embedding, init, linear, lora, moe, norm,
   quantizable_linear, sampling, sequential, two_proj_attention.
4. ~~**Quantized surface** — AWQ and Marlin are separate asks.~~ **LARGELY CLOSED 2026-07-29 — and
   this was the biggest wrong assumption in the original list. [verified]**
   `fuel-cuda-backend/src/baracuda/quant_w4a16.rs` already ships **both**: `marlin_gemm_f16` (:54) +
   `marlin_can_implement_f16` (:104), **and** `awq_gemm_f16` (:128) + `awq_can_implement_f16` (:186)
   + `AwqWeight::matmul_f16` (:444), plus `nf4_dequantize_{f16,bf16,f32}` (:204/:220/:236). GGUF
   k-quants for CPU live in `fuel-quantized` (avx / neon / simd128 / k_quants). With `qmatmul`'s
   total Q4_0 decompose (main `9d6ad291`), **the consumer's Marlin FFI is a deletion candidate and
   AWQ has a native path** — neither is a Baracuda ask.
   **Two caveats [judgment]:** existence is not performance parity — the honest test is a benchmark
   on the consumer's real shapes before deleting a hand-tuned FFI; and these are baracuda-backed, so
   **CUDA-only**. A consumer needing 4-bit on Vulkan or CPU is still in gap territory.
5. **Error type.** 49 `Error::Msg` + 28 `bail`. Mechanical, but touches nearly every coupled file.
5b. **No optimizer dtype-reconciliation pass — a real Fuel-side blocker, found 2026-07-29.** The
   dispatch failure that killed `llama-lazy` turned out **systemic, not a one-off**: the
   mixed-precision matmul a real model builds (`[F32, BF16, F32]`) has no CPU or CUDA kernel — those
   are uniform-key — and **[verified]** there is **no pass that reconciles dtypes by inserting
   casts**. Greps for `insert_dtype*` / `dtype_fixup` / `dtype_reconcil` / `insert_cast_fixups`
   return nothing, while both cited precedents exist: `insert_layout_fixups`
   (`fuel-graph/src/opt.rs:2716`) and `insert_residency_copies`
   (`fuel-dispatch/src/optimize.rs:397`). So the optimizer knows how to fix up *layout* and
   *residency* mismatches and has no equivalent for *dtype*. Diagnosed by the serving/dispatch
   session, which reports the intended fallback ("f32 matmul after a Cast") is promised in a Vulkan
   docstring but unimplemented — **[not verified here]**, I could not locate that string.
   **⚠ OPEN QUESTION raised by the fix (2026-07-29) — does `insert_dtype_fixups` honour C-5?**
   The port session reports that execution surfaced *"the numeric non-neutrality of the promoting
   cast"* — i.e. inserting a `Cast` to make a kernel available is **not** a numerics-preserving
   rewrite. If so, this is a **C-5 question**, not merely an optimizer detail: a pass that changes
   numerics to satisfy kernel availability is making a decision a consumer may have constrained.
   Under [C-5](architecture/15-consumer-contract.md), a consumer demanding bit-exactness or a
   tolerance budget must be able to *prune* that rewrite — otherwise it silently gets a different
   numerical path than the one it asked for, which is precisely what C-5 exists to prevent.
   **[not verified here]** I have not read the pass and do not know whether it is tolerance-gated,
   unconditional, or reports its intervention through C-4. Worth settling while the pass is days
   old: the same question applies to `insert_layout_fixups` and `insert_residency_copies`, which
   are value-preserving and therefore *should* be exempt — making dtype the one fixup dimension
   that is not automatically safe.

   **Consequence, scoped precisely (corrected 2026-07-29 — my first statement overreached).** This
   blocks any graph that *constructs* a mixed-precision matmul: `llama-lazy` does, because it loads
   BF16 and never casts, which is why Fuel's only runnable reference path is down. It does **not**
   automatically block every consumer: a consumer that casts weights to F32 **at load** never
   requests the missing kernel, because no matmul ever sees mixed dtypes. I asserted the port was
   "blocked behind this fix" without checking whether the port's own path would hit the gap — the
   same existence→behaviour overreach catalogued in the standing caveat, applied to a *gap* instead
   of a feature.

   **RESOLVED BY EXECUTION 2026-07-29 — F32-at-load works.** The port session built a scratch crate
   depending on Fuel by path and ran a control/test pair on `a[2,3] @ w[3,2]` with known values: the
   **control** (`F32 @ BF16`) reproduces the gap exactly, and the **test** (`F32 @ cast(BF16→F32)`)
   realizes `[4, 5, 10, 11]` — correct. A cast-at-load consumer is **verified unblocked**, not merely
   plausibly so. Gap 5b is real and blocks `llama-lazy` plus any graph that *constructs* a
   mixed-precision matmul; it does **not** block a cast-at-load consumer. Awaiting a
   build-now-vs-sequence call; the fix belongs beside the two existing fixup passes.

   **[note] The first entry in this survey established by execution rather than inspection — and it
   needed a control to be trustworthy.** The port session's harness was initially wrong and *both*
   arms failed identically; had only the test been run, the conclusion would have been "F32-at-load
   is broken" and a Fuel gap that does not exist would have been reported. **Two paths failing the
   same way is a harness smell, not a finding.**

5c. **Graph affinity is undiscoverable from the constructor side — a consumer-blocking documentation
   gap. [verified 2026-07-29]** Found by the first consumer to write a two-tensor program, by hitting
   a panic rather than by reading.

   Every `LazyTensor::from_*` constructor **mints its own graph** — `from_f32`, `from_bf16`, and
   `zeros`/`full` which delegate to them; none takes a graph parameter. Meanwhile `matmul` asserts
   `Arc::ptr_eq(&self.graph, &other.graph)` (`fuel-graph/src/lib.rs:3913`). **So the obvious way to
   write a two-tensor program — construct both, multiply them — cannot work**, and fails with
   `"matmul: tensors must live on the same graph"`: a message stating the invariant, not the cure.
   The `matmul` doc directly above that assert covers ranks, shapes and broadcasting and says nothing
   about graph affinity.

   The remedy is the **`const_*_like` family** (`const_f32_like`, `const_bf16_like`,
   `const_like_dtype`, …) — graph-sharing constructors taking an anchor tensor. `const_bf16_like`'s
   own doc names this exact case ("bf16-on-device weights in the mixed-precision matmul path —
   activations stay f32, weight matrices live as bf16"). **Clear once you know to look there;
   undiscoverable from the `from_*` side.**

   **The general constraint every consumer must internalize:** every tensor in a graph descends from
   a common root — a model's weights are `const_*_like` off the activation tensor.

   **Suggested fix, cheapest first** — deliberately *not* applied here: it spans nine assert sites in
   a core file (`matmul`, `qmatmul`, `conv2d` ×2, `conv_transpose2d`, `flash_attn` ×3) plus the
   constructor docs, and wants to land as one coherent sweep rather than a partial edit while another
   session is working those crates. **[verified]** no test matches on those message strings, so they
   are safe to change.
   1. Cross-reference `const_*_like` from every `from_*` constructor doc, and say that the
      constructor mints a **new** graph.
   2. Extend each assert message to name the cure — *"…must live on the same graph; use
      `const_*_like` to build on an existing graph"*.
   3. State the common-root constraint once in the `LazyTensor` type doc.

   **SHIPPED 2026-07-29** (`7a284021`, gate green — `fuel-graph --lib` 473 passed including both
   `should_panic` tests): type doc, five constructor docs (`from_f32`, `from_f64`, `from_bf16`,
   `zeros`, `full`), and 25 assert messages, append-only in the house style `matmul` established.

   ### Which discoverability mechanism actually reaches a consumer — and the ranking inverted

   **This is the most transferable finding in the survey, and it cost two reversals to get.**

   The consumer first reported that constructor docs would have *prevented* the failure while the
   assert would only have *shortened* it — so the work was built in that order. They then checked
   their **transcript** rather than their memory and found the account was a reconstruction:

   - for `from_f32` / `from_bf16` they ran `grep -n "pub fn from_f32\|pub fn from_bf16"`, which
     returns **line numbers only** — the doc comment was never read;
   - for `zeros` they ran `sed -n '5010,5030p'`, and line 5010 **was** `pub fn zeros(` — the window
     started *at the signature* and ran downward into the body.

   **Doc comments sit above the signature. Locate-by-grep then read-forward structurally skips
   them.** Both new constructor clauses land in precisely the region that navigation style never
   visits — and that style is typical for anyone exploring an unfamiliar 6,000-line file, which is
   exactly the situation the fix exists for.

   **The corrected ranking, by how reliably a mechanism reaches a reader:**

   1. **Runtime error messages** — arrive regardless of reading method: docs, signatures, or
      copy-paste from an example. **No blind spot.** (I had called these "the cheap half.")
   2. **Type-level docs** — at the top of the type, so they survive many navigation styles.
   3. **Per-item doc comments** — a real structural blind spot for grep-driven readers.

   **Remaining blind spot, and the consumer's own judgement on it:** only something in the *same
   visual region as the signature* would have reached them — a name or signature that says
   "graph". They explicitly did **not** propose a rename, and none is proposed here; it is recorded
   as the honest boundary of what documentation can fix. Relatedly, the panic is a **runtime**
   assert on what is really a **construction-time** mistake: the graph is already wrong when the
   second `from_*` returns, and the failure only surfaces when the tensors meet.

   **[judgment] Do not spend the next increment of discoverability effort on doc prose.** All three
   layers are correct and each catches a different reader, but the marginal return is now in error
   messages and in whether the API can make the mistake unrepresentable at all.
6. **Hot-path attention observation — the only genuine design problem in the port, and it is a
   *clause* problem.** The ~4 real extraction sites from item 1 (`custom_transformer.rs:593`/`:749`,
   `custom_attention.rs:919`, `kv_compression.rs:595`/`:820`) are all the same thing: **observing
   attention scores mid-forward**. H2O needs realized attention weights each step to accumulate
   heavy-hitter statistics; R-KV needs `attention_scores.sum(1)` realized to compute pruning
   importance.

   This is C-6 at **per-token, per-layer cadence on every request** — a regime §15 did not
   anticipate, since it framed observation around calibration, distillation, probing and debugging,
   all of which are occasional. And it collides with the thing the consumer is porting *for*:
   "naming an intermediate defeats fusion across it", while `CapturedRun` needs a stable graph with
   no host-side branching in the step. **Attention-driven eviction and captured replay appear
   mutually exclusive** on current understanding — and the consumer would discover that only after
   building on both.

   **§15 amended in response** ([C-6 §Two regimes](architecture/15-consumer-contract.md#two-regimes--and-the-obligation-differs-added-v03-2026-07-29),
   v0.3): for hot-path observation, "we reported the cost" is not a resolution. The order of
   preference is now **express the reduction in-graph** first — H2O wants a decayed running sum,
   R-KV an importance score, and neither needs the raw scores. A running accumulator across steps is
   structurally a KV write (runtime-offset write into a persistent buffer), so the machinery largely
   exists. **[judgment]** if that route holds it is a better answer than either side had: it moves
   H2O's statefulness — the capability flagged for upstreaming in A.4 — out of consumer bookkeeping
   and into the graph.

   **The consumer's spec** (`docs/superpowers/specs/2026-07-29-attention-reduction-in-graph.md`,
   Lightbulb repo) works the conjecture out concretely. The load-bearing move is a **decomposition
   the consumer's own implementation currently conflates**: *the accumulation runs every step; the
   scoring runs only when evicting.* Only the accumulation is hot. Split them and the per-step half
   is a small tensor recurrence that can live in-graph, while the on-demand half stays consumer
   policy where §15 puts it.

   H2O's exact recurrence, per step *t* and key position *k*: `a_t[k] = Σ_q attn[q][k]` (column sum
   over the query axis); `c_t[k] = decay·c_{t-1}[k] + a_t[k]`; `n_t[k] = n_{t-1}[k] + 1`. **`c` and
   `n` are the only cross-step state.** In-graph that is a reduce-sum to `[key_len]`, a persistent
   `[max_slots]` f32 buffer, and a runtime-offset read-modify-write — i.e. the
   `InferenceContext`/`Op::WriteSlice` pattern pointed at a statistics tensor, with the arithmetic
   `c·decay + a` matching the registry's existing `inplace_affine`. **[verified]** both
   `Op::WriteSlice` (`fuel-ir/src/dispatch.rs:398`) and `registry/inplace_affine.rs` exist, so
   "plausibly zero new primitives" holds for the *accumulate*. `n_t` may not need to exist at all —
   `steps_present = t − insertion_step[k]`, implied by slot position — collapsing per-step state to
   one tensor and one fused update. Everything else (`avg = c/n`, `score = 1/avg`, the sink window,
   ranking) is host-side and reads `c` once per eviction: regime 1, fine.

   **The falsifier that matters, and it is confirmed *conditionally*.** The consumer's own lead
   objection: under a tiled-softmax flash kernel the `[q][k]` matrix is **never materialized**, so
   a column-sum of it is not a reduction but a request for a different kernel. **[verified here]**
   that is true of the *fused* arm and false of the *decomposed* one — `registry/flash_attn.rs:235`
   builds `scores = scale · (q · kᵀ)` as a real node. So **observability is arm-dependent**, and
   the request is partly a C-5 arm-pruning question. §15 v0.4 records this, restates preference 2
   as **a second output from the producing op** ([12-multi-output](architecture/12-multi-output.md),
   not C-6), and notes that `flash_attn` already carries an optional `softmax_lse` in its signature
   (`:67`) — auxiliary attention statistics are an established shape for these kernels, so the
   multi-output route is a backend ask rather than an invention.

   **Falsifier #2 (accumulate-vs-capture) — RESOLVED 2026-07-29, analytically. It dissolves.**
   Three independent legs, each **[verified]** by reading:

   1. **The capture invariant is zero-alloc-on-replay, not "no runtime-varying values."**
      `baracuda/attention.rs:1386` states it exactly: two same-shape launches "must allocate the
      scratch **EXACTLY ONCE** each… proving the second launch reused the cache (**zero
      `cuMemAlloc` — the CapturedRun invariant**)." Allocation on first launch is fine; replay must
      allocate nothing.
   2. **Runtime offsets are device-resident *by design*, so they never force a rebuild.**
      `fuel-core/tests/write_slice_doff_kv.rs` — `Op::WriteSliceDoff` takes its write start from a
      rank-0 `I64` operand, "read host-side on CPU; **device-side under CUDA so a captured graph
      replays at the host-updated position**." This is the mechanism KV append already uses *inside*
      captured decode, which is why the 10.4× decode replay works at all.
   3. **In-place affine is wired.** `fuel-dispatch/src/baracuda_dispatch.rs:1959–1963` binds
      `affine_inplace_{f32,f64}` to baracuda's single-pointer kernels, so `c = c·decay + a` has a
      no-new-buffer CUDA path.

   **And the stronger result: the accumulator doesn't need the runtime-offset machinery at all.**
   If attention is computed against the *capacity-shaped* KV buffer (Fuel's existing pattern —
   fixed capacity plus a runtime valid length), the column-sum is naturally `[max_slots]`-shaped
   with masked positions contributing ~0 post-softmax. Then `a_t` is fixed-shape, `c_t =
   c_{t-1}·decay + a_t` is a fixed-shape in-place elementwise affine, and there is **no dynamic
   extent, no slice, and no offset anywhere in the recurrence**. It allocates `c` once and reuses it
   forever, which satisfies the capture invariant trivially rather than narrowly.

   **What this does not establish.** Nothing here was executed. The decisive test is the existing
   pattern — build the `[max_slots]` accumulator, launch twice, assert `allocation_count == 1`,
   exactly as `fused_rope_is_capture_safe_zero_alloc_on_reuse` does — and it requires a live CUDA
   device (the invariant is `cuMemAlloc`; CPU cannot answer it). **Falsifier #1 (arm-dependence) is
   untouched and remains the real blocker**: this result says the accumulator is capture-safe *given*
   `a_t`, not that `a_t` is obtainable on the fused arm.

   **Correction to the fixed-shape form — it drops reset-on-reuse (raised by the port session,
   2026-07-29).** The capacity-shaped formulation above silently loses a correctness requirement the
   consumer's `HashMap<slot_id, TokenMetadata>` form gets for free: when a slot's occupant is
   evicted and the slot is reused, the map entry *vanishes* and the next occupant starts fresh. A
   fixed-shape `[max_slots]` buffer has no removal, so `c[k]` keeps decaying and a new token
   inherits its predecessor's remnant. **The `n` direction is the worse half**: a reused slot would
   report the *previous* occupant's tenure, inflating the denominator in `avg = c/n` and making a
   brand-new token look long-lived and low-attention — precisely the profile H2O evicts first, so a
   freshly-admitted token could be immediately re-evicted.

   **Their fix preserves every property established above**: an occupancy mask,
   `c_t = (c_{t-1}·decay + a_t) ⊙ occ_t`, with `occ_t` a `[max_slots]` 0/1 vector zeroed on
   admission. Still elementwise, fixed-shape, in-place-able, **still no offsets and no dynamic
   extent** — the zero-`cuMemAlloc` argument is untouched. It makes the op `c·decay·occ + a·occ`
   rather than a bare `InplaceAffine`, which lands in the same bucket as falsifier #3: *which op*,
   not *whether capture holds*. And it **sharpens the `n_t = t − insertion_step[k]` collapse** —
   `insertion_step` is exactly what must be reset on reuse, so one `[max_slots]` buffer written on
   admission yields `n_t` by derivation *and* `occ_t = (insertion_step ≥ 0)`. One buffer solves both.

   **[verified here, with a refinement neither of us had]** `h2o_policy.rs:209`'s `clear_slot` is
   real and is called — but at `parallel_cache_builder.rs:1909` it sits inside
   `reset_batch_index(batch_index)`, which zeroes `positions`/`indices` for a **batch row**: that is
   *sequence-level* reuse (a new request taking over a row), not *token-level* eviction within a
   sequence. The token-level path is `should_clear_slot` → KV-cache `clear_slot()`, and
   `parallel_cache_builder.rs:2048` says plainly: **"CURRENT STATUS: Not actually used yet since
   `clear_slot()` is a stub."** So token-level slot reuse is **not live in the consumer either**.
   The mask is therefore not repairing a regression the fixed-shape form introduced — it implements
   a requirement **neither form meets today**.

   **[judgment] The architectural consequence is the interesting part, and it is not H2O-specific.**
   Token-level slot reuse becomes live exactly when Fuel's block-pool allocator lands, because
   refcount-aware evict plus block reuse *is* that mechanism. So **any per-slot side buffer built
   above the allocator silently inherits stale state unless slot-recycle is observable at the
   allocator boundary** — either the allocator zeroes registered per-slot side buffers on recycle,
   or it surfaces a recycle signal a consumer can hook. That is a mechanism obligation (slot
   lifecycle is the allocator's), it generalizes past H2O to every per-slot statistic, and it is
   worth settling while the evict/restore surface is still being designed. Routed to the allocator
   session as part-2 input; **not** promoted to clause text, since it is design-level and unvalidated.

   **Falsifier #3 (per-slot decay) — downgraded, not resolved.** If decay must vary per slot, the
   scalar `mul` becomes a vector and the op is no longer `InplaceAffine` — but it is still
   fixed-shape, still elementwise, and still in-place-able. That changes *which op*, not whether
   capture holds. (The mask term above lands in the same bucket, and composes with it.)

   Open on the consumer side: whether R-KV's attention input is captured per-step (regime 2) or
   only at compression time (regime 1) — flagged rather than guessed.

   **Open, and it bears on whether the consumer's H2O is a usable reference at all: a possible
   index-space confusion.** `H2OPolicy::update_attention_scores` populates `slot_metadata` keyed by
   **`slot_id`** (from a `cache_positions: HashMap<slot → seq_position>`), while
   `reset_batch_index(batch_index)` calls `clear_slot(batch_index)` keyed by **batch row**. If those
   two index spaces are not identical, `clear_slot` clears the wrong entries — H2O metadata for a
   finished request would persist while an unrelated slot's history is discarded. They may coincide
   (one KV row per batch slot, which the `[max_batch, heads, seq, head_dim]` cache shape hints at).
   **Raised by the port session as a question, explicitly not a claim, and unresolved.** Recorded
   because if the spaces diverge, the existing reset behaviour is not merely incomplete but *wrong*,
   and **the in-graph form must not inherit its structure**.

   **Incidental drift found while verifying:** `registry/inplace_affine.rs`'s module doc says
   "Backend dispatch (CPU + CUDA `affine_inplace_*`) lands in **Phase 3**… until then, the
   metadata-side entry exists so CSE, telemetry and dispatch work." CUDA dispatch **is** wired
   (`baracuda_dispatch.rs:1959`), so that comment is at least partly stale. Not corrected here — the
   CPU half is unverified, and correcting it needs someone to confirm both.

   **Status: conjecture, not result.** Derived by reading real consumer code against the clause and
   against `CapturedRun`'s requirements; not validated by a running port. Validation order:
   confirm the attention matrix is materializable on the arm the consumer needs → express the
   recurrence as graph nodes → confirm `CapturedRun` still captures the step. If step one fails,
   §15's preference 4 (state the incompatibility plainly) is the honest fallback.

**Finding for serving Increment 2 [judgment].** Lightbulb will almost certainly **not** adopt
`SessionScheduler` — it has its own scheduler, slot pool, and admission. It wants the **allocator**
underneath its own policies. That validates deferring the scheduler-surface reshape and may shrink
that reshape permanently. Two allocator requirements Lightbulb's cache implies:

- **Prefix sharing** — `prefix_cache.rs` + `cache_span.rs` mean shared-prefix blocks across sessions,
  which is exactly the refcounted COW splice being built. Good fit, no change needed.
- **Compressed KV** — `kv_compression.rs` is 1,998 LOC, so block sizing may not be uniform across
  sessions. **[judgment]** worth confirming the allocator doesn't assume fixed-size blocks in a way
  that forecloses compressed-KV consumers.

### A.4 `fuel-inference` overlaps the consumer heavily (2026-07-29)

Found while verifying a port question; post-dates A.3 and materially changes the port's shape.

**[verified]** `fuel-inference` is 6,231 LOC across 14 modules and ships: eviction policies (LRU,
H2O heavy-hitter, weighted voting), prefix caching, StreamingLLM, speculative decoding, chunked
prefill, segmented eviction, KV compression (KIVI / R-KV / low-rank), a memory-aware scheduler with
priority queuing and eviction-pressure admission control, MoE routing, tiered storage (GPU→CPU→Disk
with RoPE re-injection), context compression, tool call, sampling + `LogitsProcessor`, and unified KV
cache variants.

Set against Lightbulb's tree the **module names** line up closely: `cache/{h2o_policy, prefix_cache,
streaming_policy, segmented_eviction_policy, kv_compression, tiered_storage, eviction_policy}.rs`,
`model/chunked_prefill.rs`, `engine/{moe_router, speculative, memory_aware_scheduler,
context_compression, tool_call}.rs`, `sampling.rs`.

> ### ⚠ A.4's original framing was wrong — corrected 2026-07-29
>
> This annex originally called that a **"near-1:1 overlap"** and implied the port's main outcome
> would be the consumer deleting ~9.4k LOC. **On execution-grade evidence that inverts.** The
> overlap is real at the **module-name** level and substantially weaker at the **capability**
> level. Verdicts below are the port session's, from reading both implementations.
>
> **Final tally, all 13 modules diffed: 1 clean adopt, 2 name collisions, 6 compose/complementary,
> 2 judgment calls, 1 resolved earlier (`sampling` → consumer policy, Q5).**
>
> **⚠ These verdicts are hypotheses, not decisions (re-labelled 2026-07-29).** The diff established
> *"Fuel has X"* for all thirteen modules and *"Fuel's X does what the consumer needs"* for **none**
> of them — the first is a source read, the second requires running. "Adopt" and "compose" below
> read as conclusions; they are a **shortlist of what to test**. Treat every row as a port decision
> still pending, and see the standing caveat at the head of A.3 for why this distinction has been
> the session's dominant failure mode.
>
> **Not overlap at all — pure name collisions:**
> - **`tool_call`** — Fuel's is schema + registry + *post-hoc text* parsing (`ToolDef`,
>   `ToolRegistry`, `validate`, `extract_tool_calls(text)`). The consumer's is a
>   `ToolCallDetector`: *token-level streaming* detection during generation (`push_token`,
>   `AttentionSnapshot`, per-cache-slot state). Different capabilities; adopt Fuel's registry,
>   keep the detector.
> - **`engine/streaming_context.rs`** — "Streaming Context *Injection*"
>   (`StreamingContextProvider`, `ContextStream::on_token`, code-completion and web-search
>   providers). Nothing to do with Fuel's `streaming.rs` StreamingLLM sink tokens. *(The port
>   session's own mis-pairing, corrected by them.)*
>
> **The five genuine overlaps — only one is a clean adopt:**
>
> | Module | Verdict |
> | --- | --- |
> | `streaming_policy` | **adopt Fuel's** — it has `position_ids` for RoPE remapping + `select_keep`/`select_evict`; the consumer's is index arithmetic |
> | `kv_compression` | **compose** — adopt Fuel's `CompressedKv`/`KvCompressor` traits; **upstream** KIVI granularity (`QuantGranularity::{PerHead, PerGroup}`, `per_head_scales` — Fuel's `KiviConfig` has *only* `bits`) and a relationship-aware strategy with no Fuel counterpart |
> | `speculative` | **compose** — Fuel's `verify_draft` + stats is a verification *primitive*; the consumer's is a *driver* (`SpeculativeModel` trait, `generate_tokens`). Fuel has no loop; the consumer has no standalone verify. Splits on the mechanism/policy line exactly |
> | `eviction` + `h2o` | **compose** — adopt Fuel's `EvictionContext`/trait/`VotingAggregator` (`Box<dyn>` beats a generic builder); **upstream stateful H2O** — Fuel's `H2oPolicy` is a unit struct scoring a passed-in snapshot, the consumer's accumulates `TokenMetadata` (cumulative attention, steps present, position) across steps with a decay factor |
> | `prefix_cache` | **adopt Fuel's core** (`longest_prefix_match`); keep the consumer's observability (`hit_rate`, `avg_saved_tokens`, `current_size_bytes`, `check_would_hit`) |
>
> **The remaining six:**
>
> | Module | Verdict |
> | --- | --- |
> | `scheduler` | **judgment call, not capability** — near-identical concepts; the difference is coupling (Fuel's is standalone, the consumer's extends its own `SlotPool` + `slot_monitor`, ~1k LOC). Both are policy, so genuinely optional per [§15 v0.2](architecture/15-consumer-contract.md#where-the-seam-runs-foundation-not-the-repository) |
> | `chunked_prefill` | **complementary** — Fuel chunks *one* sequence (zero-copy `ChunkedPrefill<'a>::next_chunk`); the consumer batches *across* requests (`next_batch(&mut [PrefillRequest])`) plus tensor materialization |
> | `tiered_storage` | **complementary** — Fuel = tier accounting + demotion policy; the consumer = byte movement + storage backends |
> | `segmented_eviction` | **compose** — Fuel has the span vocabulary; the consumer adds parent/child hierarchy with cycle detection, `importance: f32`, and partial-vs-full `EvictionImpact` |
> | `moe_routing` | **compose — a C-4 instance** *(verdict revised 2026-07-29, upward)*. Fuel's `RoutingResult` has `num_dropped` + `expert_load`, so it reports the *symptom*; the consumer adds `load_imbalance()` + `RoutingStats`, the metric that makes it **diagnosable** — it tells a consumer `capacity_factor` is wrong *before* tokens start dropping. That is C-4's "advertised cost is a hint; measured cost is the record" applied to expert load, so it upstreams on the same footing as the others, not as polish |
> | `sampling` | **resolved earlier** — consumer policy (Q5) |
>
> **Three findings from the completed diff worth pulling out:**
>
> **(a) `scheduler` is NOT a stub — my §15 worry was wrong, in Fuel's favour. [verified here]**
> `MemoryScheduler` (`scheduler.rs:139`) has real admission — `try_admit(RequestInfo) ->
> Option<SlotHandle>` (`:185`), a configurable `pressure_threshold` (`:150`, `:172`), plus
> `drain_queue`, `update_usage`, and budget accounting. I had asked whether "optional
> consumer-side toolkit" was a generous description of a stub. It isn't generous; it's accurate.
>
> **(b) `tiered_storage` is metadata-only *by explicit design*, and it was written against a
> mechanism that did not exist. [verified here]** `tiered_storage.rs:6` — it "does not move actual
> tensors (**that responsibility belongs to the caller / runtime**)." `TieredStore` tracks tiers
> and budgets and returns `TierTransfer` *descriptors* (`:109`, `:258` `candidates_for_demotion`).
> **The "caller/runtime" it defers to is precisely C-3's evict/restore** — so this is a
> C-3-lossy consumer authored before C-3 existed, waiting for the allocator to be its byte-moving
> half. Strongest available input to the evict/restore signature.
>
> **(c) Fuel's toolkit already ships a span vocabulary. [verified here]** `segmented_eviction.rs`
> carries `SpanId` (`:60`), `SpanKind` (`:79`), `SpanInfo` (`:109`), `EvictionPlan` (`:141`), and
> `SpanRegistry::register(label, kind, range) -> SpanId` (`:181`, `:206`) — spans described as
> "named indivisible units (a system prompt, a conversation turn, a retrieved document chunk)."
> **This matters to the allocator's deferred "named, refcounted block group" increment**, which is
> currently framed as greenfield to be designed against two consumers: one of those consumers
> already ships the abstraction. It re-poses the question from *what should a span be* to
> *should the toolkit's `SpanRegistry` sit on a Foundation-level group handle* — better-posed, and
> only findable by diffing.
>
> **[judgment]** (b) and (c) are the same shape as A.4's headline lesson: `fuel-inference` is not
> a duplicate of a consumer, it is **a policy layer written against Foundation mechanisms that
> were never built**. That is why it has 153 tests and zero consumers.
>
> **The dominant outcome is upstreaming, not deletion.** On current evidence Fuel would *gain*
> stateful H2O accumulation, KIVI granularity control, and a relationship-aware compression
> strategy. The size deltas flagged below (`kv_compression` 1,998 vs 742) turn out to be **real
> capability, not padding** — exactly the case CireSnave's rule sends upstream.
>
> **Why this correction matters more than it looks** *(the port session's framing, and it is the
> right one)*: the harm of the original wording was never that a consumer deletes something it
> shouldn't. It is that **it makes Fuel look complete where it isn't.** A name-level inventory
> read as a capability inventory turns "Fuel has a module called X" into "Fuel does X," and that
> error propagates into roadmap decisions no consumer is present to correct.

**Maturity, precisely [verified — sharpened by the Lightbulb session]:** 153 `#[test]` fns across the
12 modules plus `tests/scheduler_bridge.rs`, and **exactly zero consumers** — `fuel-inference`
appears in only two `Cargo.toml` files, the workspace root and its own. So it is **unit-tested but
never integrated**. The first consumer should expect to find the integration bugs 153 unit tests do
not catch. *"Tested" must not be read as "proven."*

**Decision rule (CireSnave, relayed 2026-07-29):** default is **adopt Fuel's** — "unless Lightbulb
has some piece of functionality worth keeping over Fuel in spots where they are nearly 1:1, we should
use what Fuel supplies." And upstreaming is blessed: "if Lightbulb does contain functionality that
would be better suited on the Fuel side of that seam, we can move things into Fuel." So the
per-module diff has three arms — **adopt Fuel's**, **upstream Lightbulb's**, or **diverge with a
stated reason** — with the burden of proof on keeping a consumer-side copy.

Size deltas that make this actionable rather than academic: Lightbulb's `kv_compression.rs` is 1,998
LOC vs. `fuel-inference`'s 742; `segmented_eviction_policy.rs` 844 vs. 551. KIVI / R-KV / low-rank
are named on both sides, so the delta is plausibly depth rather than breadth — **[judgment]** if it
is real capability, the answer is upstream, not keep.

**Note the placement question this raised**, now resolved in the constitution: `fuel-inference`
shipping admission control and eviction *policy* looked like a violation of §15's refusals. It isn't
— see [15 §Where the seam runs](architecture/15-consumer-contract.md#where-the-seam-runs-foundation-not-the-repository)
(added v0.2 in response): the seam runs between Foundation and the orchestration tier, not around the
workspace, and `fuel-inference` is an optional consumer-side toolkit above it.

---

## Annex B — Training host

**Unit** an optimizer step, or a microbatch under gradient accumulation. **State** parameters +
optimizer moments + **RNG stream position**, exactly-restorable. **Cost** samples/sec, step time,
activation and gradient bytes.

**What is the consumer's:** optimizer choice, LR schedule, batch size, response to a NaN or diverged
step, checkpoint cadence, early stopping. All are *different math* or *value judgments*, so the rule
assigns them without ambiguity.

**What is Fuel's:** gradient checkpointing (recompute-vs-store is the same gradients at a different
memory/time point — arm selection under a consumer-supplied memory budget), mixed-precision arm
selection within a consumer-supplied tolerance, and sharding across a consumer-supplied device set
([06-runtime](architecture/06-runtime.md) already owns data parallelism).

**Clause specialization.** C-1: activation-memory headroom, which is what lets the consumer choose
microbatch size vs. gradient accumulation — Fuel says what fits, the consumer picks. C-2: matters for
spot-instance preemption, elastic training, and sharing a device with an inference consumer; quantum
= a step or microbatch. **C-3: this is durability, not load-shedding** — restore must be exact and
*must include the RNG stream position*, or a resumed run silently diverges from an uninterrupted one.
C-4: unchanged in shape. C-5: the strongest case in the taxonomy — a reproducibility requirement
prunes every non-deterministic arm, and a mixed-precision tolerance bounds which numerical arms are
legal.

**Open:** whether a trainer's C-3 and the inference C-3 share one mechanism with a fidelity flag, or
are genuinely two implementations. **[judgment]** one mechanism, two guarantees — but this should be
settled by whoever builds the second one, not asserted here.

---

## Annex C — Stateless batch (embeddings, retrieval, quantized-only paths)

**Unit** one batch. **State** none between calls. The degenerate case, and useful precisely because
it shows the contract *degrading gracefully*: this class needs C-1, C-4, and C-5 and nothing else.

C-3 is not applicable — there is no engine-held state to externalize. C-2 is optional; batches are
short, though cancellation still matters for large ones. C-5 carries real weight here: quantized-only
consumers are explicitly trading numerics for throughput, so the tolerance budget *is* the interface.

If a future clause cannot be omitted for this class, that is a signal the clause is really two
clauses.

---

## Annex D — Oracle / test runner

**Unit** one reference execution. **State** none. Throughput is irrelevant; correctness is
everything.

**C-5 is essentially the entire contract for this class** — require bit-exactness, forbid
non-deterministic arms, forbid approximate rewrites. That makes this class the natural *test* for
C-5: if a consumer can demand determinism and verifiably get it, C-5 works. It also connects to the
existing calibration discipline (sabotage-calibrated epsilons) and to the kiss-ref advisory
cross-check, which is an oracle consumer in all but name.

C-4 is still wanted — measured cost feeds the ledger and calibration even when the runner does not
care about speed.

---

## Annex E — IPC / remote (mlmf) — stub

Named because the ROADMAP already names it ("inter-process tensor exchange (Fuel ↔ Lightbulb ↔ mlmf
using safetensors as the wire schema)", `RemoteHostStorage` Phase 7c). The wrinkle is that state
crosses a process boundary, so C-3 becomes *serialization* rather than eviction and overlaps
[13-interchange](architecture/13-interchange.md). Left as a stub deliberately — writing it without a
consumer would be speculation.

---

## Annex F — RL / alternating train-infer loop (GRPO, RLVR, online learning)

**Unit** a rollout batch, then a policy update. **State** *both* kinds at once — KV caches from
rollouts (lossy, discardable) and parameters + optimizer moments + RNG stream (exact). **Cost**
rollout tokens, update step time, and the ratio between them.

This class matters out of proportion to its size because it is the **only** one that runs A and B
against the same weights with a mutation between phases. Every other class holds weights still for
its entire lifetime. Already on Fuel's roadmap (GRPO/RLVR, bucket D in
[`frontier-paradigms-vision.md`](frontier-paradigms-vision.md)), so this is a scheduled consumer, not
a hypothetical.

**Why it is the C-7 consumer.** After a policy update, everything derived from the old weights is
stale: rollout KV caches (semantically — they encode the old policy's continuations), captured runs,
and any arm choice or cost estimate conditioned on the old weights. The consumer knows the mutation
happened; Fuel does not, unless told. Nothing in the current surface provides the telling, and the
failure mode is silent rather than loud — a stale capture replays, and the run degrades instead of
erroring.

**Other clause pressure.** C-3 must serve both fidelities in one process, which is the strongest
argument for one mechanism with a fidelity flag rather than two implementations (Annex B's open
item). C-5 is doubly binding: rollouts and updates may legitimately want *different* determinism and
tolerance settings, so constraints must be per-phase, not per-process. C-2 matters because rollout
batches are long and the consumer wants to bound them against update latency.

**[judgment]** If C-7 is going to be built, this is the consumer that should specify it, and it
should be specified before the inference-side C-3 hardens — otherwise C-3 gets an inference-shaped
API that this class cannot use.

---

## Annex G — Observation / analysis (calibration, distillation, interpretability)

**Unit** one instrumented execution. **State** none. **Defining clause** C-6; without it this class
cannot exist at all, which is why it earns a row despite being clause-light everywhere else.

Quantization calibration reads activation statistics to compute scales. Distillation reads teacher
intermediates. Interpretability reads, patches, and ablates. Debugging — every consumer's fallback
mode — reads whatever is going wrong. These differ in what they do with the value, not in what they
need from Fuel.

C-5 is joint-binding with C-6 here: an observation is only meaningful if the consumer also controls
the numerics that produced it. Calibrating scales against an ε-close fused arm and then deploying
against a bit-exact one (or the reverse) is a silent correctness bug, and the consumer needs to be
able to pin both.

**[judgment]** This class is the natural first consumer of C-6 because calibration is concrete,
already needed by the quantized paths Fuel supports, and produces a checkable artifact — which makes
it a better proving ground than interpretability, whose requirements are open-ended.

---

## Relationship to the Reasoning-Runtime sketch

Context only; nothing above depends on it. Under a "cognitive engines as schedulable resources"
framing, the layering is nested runtimes, each the same shape, each with a less fungible cost unit
than the one below. **Fuel is not an engine** — it is the substrate an engine is built from (Candle's
former slot). An inference host like Lightbulb is externally *one* engine and internally a runtime;
the engine boundary is the **unit of resource arbitration**, not the unit of capability, so several
models sharing one VRAM pool are one engine with a union capability profile, not three engines.

A *trainer* is not an engine in that framing at all — it serves no requests. That is the sharpest
reason the contract stays consumer-agnostic: **Fuel must not assume the inference shape**, because at
least one first-class consumer class doesn't have it. (The constitution now says this directly —
[15 §Consumer-shape neutrality](architecture/15-consumer-contract.md#consumer-shape-neutrality) —
after the 2026-07-28 withdrawal of `09`'s inference-center-of-gravity claim.)

The consequence is that C-1…C-7 are not speculative infrastructure for a hypothetical layer. They are
what multi-session serving and reproducible training each need on their own merits.

---

## Open questions

1. ~~Promote the core to a numbered constitution section?~~ **Done 2026-07-28** —
   [15-consumer-contract](architecture/15-consumer-contract.md) v0.1, with `09-non-goals` v0.4
   correcting the inference-center-of-gravity claim. See
   [10-decisions-log §2026-07-28](architecture/10-decisions-log.md).
2. **Does `multi_session.rs` move to `fuel-inference`, or does the layer table get an argued
   exception?** Recommendation: move, before the port. (Annex A.2 — verified defect.)
3. ~~Is C-3 in scope for Increment 2?~~ **RESOLVED 2026-07-29 (cae56435) — YES, and it IS the
   block-pool allocator.** Paged blocks + refcounting *are* the evict/restore/splice mechanism; C-1
   falls out of the same free list. One coherent piece. The lossy-KV arm shipped as the allocator
   core (`fuel-core/src/kv_block_pool.rs`, part 1); the training/RL exact fidelity is a later
   increment (see Q9). Design:
   `docs/superpowers/plans/2026-07-29-kv-block-pool-allocator-serving-inc2.md`.
4. **Rename `SchedulePolicy` → `DecodeArm`?**
5. ~~Where does sampling live?~~ **RESOLVED 2026-07-29 (Annex A.3) — consumer policy.**
   `Lightbulb/src/sampling.rs` is host-side post-processing over *realized* logits, with
   constrained generation layered on top in `contracts/`. Fuel should produce logits and stop;
   `SessionState::sample_and_append` is misplaced and rides the `fuel-inference` move. Not an eighth
   clause. Sub-finding bounds C-3-exact: the consumer owns its sampler's RNG, so the "RNG stream
   position" an Exact handle must cover is *Fuel's* RNG, never the consumer's.
6. ~~What is the minimum a consumer needs to port at all?~~ **RESOLVED 2026-07-29 (Annex A.3) — the
   clauses do not gate the port.** Lightbulb keeps its own engine, cache policies, memory, slot pool,
   scheduler, sampler and API; the port is a tensor-layer swap concentrated in `model/` (16 of 17
   files), not a re-architecture around this contract. C-1…C-7 are adopted incrementally. The real
   gates are Lightbulb-side (eager→lazy, 70 value-extraction sites) plus a Fuel-side parity list —
   `fuel-transformers`, `fuel-nn` coverage, the quantized surface (AWQ, Marlin), and the error type.
   See the reverse gap list in A.3.
7. **Does C-6 land as Phase-9 `RuntimeHook`, or a separate seam?** The hook machinery is specified
   but unbuilt (**[verified]** — it appears in `ROADMAP.md` and two frontier docs, and in zero source
   files); if it is built for constraint-gating without the consumer-facing observe/intervene
   surface, C-6 will need a second mechanism later. Cheaper to design once.
8. **Should C-7 be built now or specified now and built with Annex F?** It is undefined in a
   direction that fails *silently*, which argues for at least a loud interim: a declared
   `weights_mutated()` that conservatively invalidates everything derived, tightened later.
9. ~~Is one C-3 mechanism with a fidelity flag right, or two implementations?~~ **RESOLVED 2026-07-29
   (cae56435) — one interface + a `Fidelity` discriminator, two implementations backed by different
   state; the lossy-KV arm now.** The load-bearing rule (sharper than "fidelity flag"): `restore`
   takes externalized *state* (an opaque handle), NEVER a "recompute-from-tokens" instruction — that
   is what keeps the future `Fidelity::Exact` (training/RL: params + optimizer moments + RNG stream
   position) arm expressible. The Exact-arm completeness gate is specified now (restore diverges from
   an uninterrupted run by exactly zero; the handle enumerates its coverage; RNG coverage bounded to
   **Fuel-owned** RNG, never the consumer's sampler — per Q5), gated on the RNG/generator seam.
