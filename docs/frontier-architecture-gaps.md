# Fuel — frontier-architecture gap catalog (2026-07-04)

**Purpose.** A durable, single-place capture of the capabilities Fuel needs to run the
2025–2026 ML research frontier, so none is forgotten. This is a cross-cutting *backlog
index*, not a program plan — it sits **behind** the active frontier in
[`ROADMAP.md`](../ROADMAP.md) and anchors every item to its real home (a phase, a design
doc, a decisions-log entry) or flags it as an orphan with no prior home.

**Authority.** The constitution ([`docs/architecture/`](architecture/00-index.md)) wins
over this doc; the ROADMAP owns *sequencing*. This catalog only *tracks* — it schedules
nothing on its own. When an item here gets a phase, update the "Home" column to point at
it.

**Companion.** This catalog covers the *transformer-adjacent* frontier. For the paradigms
**beyond** autoregressive transformers — energy-based / Hopfield associative memory,
neurosymbolic execution, symbolic regression & KAN, gated compute orchestration, and
tensor-train compression — see [`frontier-paradigms-vision.md`](frontier-paradigms-vision.md),
which classifies them into the same recipe/encoding/pass/leaf buckets and records two crux
decisions (a unified bounded `Scan` primitive; a backend-like discrete-solver interface).

**Origin.** A frontier-readiness audit (six-track codebase+docs sweep on 2026-07-04)
against a survey of the current research edge: hybrid State-Space/Transformer
architectures, Multi-head Latent Attention & QKV pruning, hyper-sparse Mixture-of-Experts
& soft routing, test-time compute (inference-scaling / search-on-generation), and
GRPO / verifiable post-training. The recurring finding: **Fuel usually has the
*expressible* form (often the actual model), but not yet the *efficiency payoff*** — and
three of the five payoffs bottleneck on the same missing capability.

---

## The keystone: data-determined dynamic shapes

Three of the five frontier payoffs — **SSM autoregressive decode**, **MoE sparsity**, and
**MLA's compressed KV cache** — all gate on the *same* capability: **data-dependent
(a.k.a. data-determined) dynamic shapes over a fixed-capacity buffer + a runtime scalar
count.** The substrate is the `SymId` / `SymEnv` / `DynScalar` / `Extent` machinery.

- **Input-determined** shapes (the caller knows the bound up front — e.g. the KV attended
  length in decode) are **SHIPPED** — this is what makes plan-once persistent decode work
  (Phase D, [`docs/session-prompts/symbolic-extents-and-persistent-decode.md`](session-prompts/symbolic-extents-and-persistent-decode.md)).
- **Data-determined** shapes (the *op itself* produces the count mid-pass — MoE per-expert
  token counts, `NonZeroIndices` active-row counts, data-dependent top-k) are **DESIGNED,
  NOT BUILT** — explicitly the "recorded future" in the symbolic-extents design and the
  subject of [`docs/session-prompts/data-dependent-shapes-design.md`](session-prompts/data-dependent-shapes-design.md) (status: *design / not started*).

**This is the single highest-leverage unlock in this catalog.** Finishing the
data-determined half turns dense MoE into sparse MoE and gives the SSM/attention decode
paths their capacity-buffer machinery. Phase 8.5 already needs the identical primitive
(`Op::NonZeroIndices`) for activation sparsity, so the primitive has a second consumer.

---

## Legend

- **Status** — `Built` (works today) · `Partial` (expressible/modeled but payoff missing)
  · `Designed` (design doc exists, unbuilt) · `Absent` (no code, no plan).
- **Home** — where it is (or now is) tracked. `orphan → this doc` means it had **no**
  planning-doc home before this audit and is now tracked here (and registered in the
  ROADMAP Deferred backlog).

---

## 1. Hybrid State-Space / Transformer (Mamba, Bamba, Zamba2, Kimi-Linear)

**Built today:** the three SSM fused ops (`CausalConv1d`, `SelectiveScan`, `SsdChunkScan`)
with CPU kernels; lazy **prefill-only** ports of Mamba-1/2, RWKV-v5, and two true hybrids —
LFM2 ([`fuel-core/src/lazy_lfm2.rs`](../fuel-core/src/lazy_lfm2.rs), attention/short-conv
interleave = the Bamba/Zamba2 pattern) and Based ([`fuel-core/src/lazy_based.rs`](../fuel-core/src/lazy_based.rs)). The interleave-every-Nth-layer structure is trivial once the
ops run. **The payoff (long-context decode throughput / KV compression) is unrealized.**

| Gap | Status | What's needed | Home |
| --- | --- | --- | --- |
| **Higher-order `Scan` `Op`** (G3 basis gap) | **SHIPPED 2026-07-15** (Op::Scan Phase 1) | Built as `Op::Scan{body,carry,bound,emit,early_exit}` + the `Op::ScanPlaceholder` body-hole leaf — a bounded `lax.scan`-shaped primitive whose body lives in the node's own `inputs`. `selective_scan` + `ssd_chunk_scan` `decompose` now emit `Op::Scan` (parity-gated), **closing G3** — `decompose` is total over genuine primitives. Basis closure only: the fused SSM kernels stay the executed path; `Op::Scan` has **no native kernel** yet. | Shipped per [10-decisions-log 2026-07-15](architecture/10-decisions-log.md). **Phase 2** = early-exit mechanism + BPTT differentiability + a Hopfield consumer, and wire/drop the slot-1 (`last_state`) view **before** adding an `Op::Scan` kernel (silent-OOB otherwise) |
| **SSM autoregressive decode** | Absent | `Op::SelectiveScanWithInitState` (or a 6-input `selective_scan`) that consumes the prior step's `last_state`. Producer half exists (kernels already emit `last_state` as bundle slot 1); the consumer feedback is unbuilt. Blocks the Mamba/LFM2 decode loop — the actual KV-compression win. | orphan (source comment `lazy_mamba.rs:25-31`) → this doc |
| **GPU scan dispatch** | Absent | Wire the already-ported baracuda CUDA mamba kernels (`fuel-cuda-backend/src/baracuda/mamba.rs`) to `OpKind` dispatch (+ autograd). Today only `causal_conv1d` + `cumsum` dispatch on CUDA; the two scans do not. | orphan (source comment `mamba.rs:16-19`) → this doc |
| **GraniteMoEHybrid Mamba branch** | Partial (attention-only) | The Mamba branch currently bails; lands downstream of the `Scan` op + SSM decode above. | orphan (source comment `lazy_granitemoehybrid.rs:6-8`) → this doc |

*Sequencing:* the `Scan` op (the root) **shipped 2026-07-15** (`Op::Scan` Phase 1, G3 closed); SSM decode + GPU dispatch follow; hybrid-model
completion is downstream of all three.

---

## 2. Multi-head Latent Attention & QKV pruning (DeepSeek MLA, two-projection)

**Built today:** DeepSeek-style **MLA is fully implemented** as a lazy DAG from generic
primitives ([`fuel-core/src/lazy_deepseek2.rs`](../fuel-core/src/lazy_deepseek2.rs)) —
real low-rank latent-KV math — which proves attention is *composable*, not a rigid fused
op. GQA/MQA are first-class; `flash_attn` is an *optional* fused accelerator that
decomposes to `matmul→mask→softmax→matmul`. So the attention *math* for these variants is
free. **The cache-compression payoff is what's missing.**

| Gap | Status | What's needed | Home |
| --- | --- | --- | --- |
| **MLA decode-time compressed KV cache** | SHIPPED (D1 persistent) | Per-pass: `forward_with_latent_cache[_absorbed]` (bit-exact / 1e-6-calibrated vs one-shot; prefill-friendly). **Persistent: `forward_with_latent_kv_context`** — the D1 house pattern (fresh graph per call, device-resident `LatentKvCache`, `write_slice_dyn` at the `SymId`-bound `cached_len`, full-capacity masked reads, absorbed attention) — the decode prefix never round-trips through the host. Sabotage-calibrated parity (genuine ~1.5e-8, corrupted-offset signal ~7.7e-3, tolerance 1e-6). Remaining follow-ups (not gaps): D2 plan-once `DecodeSession` for MLA, a generate-loop driver, device-general embed bootstrap. | `lazy_deepseek2.rs::forward_with_latent_kv_context` → this doc |
| **KV-cache container generalization** | SHIPPED (both halves) | Per-forward-pass: **`LazyLatentCache`** (`lazy_latent_cache.rs`) — per layer, an ordered list of latent buffers `[max_seq, …arbitrary trailing]`, graph-anchored functional append. Persistent: **`LatentKvCache`** (`inference_context.rs`) — same N-slot generalization over device-resident `Arc<RwLock<Storage>>` buffers surviving across graphs, mirroring `KvCache::with_capacity`'s allocation + version contracts. Standard K/V = two equal slots; MLA = two *unequal* slots; two-projection = one slot. `KvCache` itself untouched (Llama/Phi keep their path); re-expressing it over the N-slot container is optional cleanup, not a blocker. | `lazy_latent_cache.rs` + `inference_context.rs::LatentKvCache` → this doc |
| **MLA weight-absorption** | SHIPPED (per-pass) | `DeepSeek2Model::forward_with_latent_cache_absorbed` folds `kv_b_proj`'s per-head `W_UK`/`W_UV` into the query/context math (`q_absorbed = q_nope·W_UK^T` attends directly against the cached latent; `ctx = (probs·c)·W_UV`) — no per-step re-projection of the prefix. Parity vs the non-absorbed path + one-shot at 1e-6 (sabotage-calibrated: a wrong `W_UK` moves tiny-fixture logits only ~3e-5, so looser tolerances mask real bugs) + per-row argmax equality. Seq-length-based switching between the absorbed/non-absorbed siblings is a follow-up. | `lazy_deepseek2.rs::forward_with_latent_cache_absorbed` → this doc |
| **Two-projection attention / QKV pruning** | SHIPPED | `LazyTwoProjAttention` (`fuel-core/src/lazy_nn/two_proj_attention.rs`) — the shared-K=V variant of "Do Transformers Need Three Projections?" (arXiv 2606.04032, ICML 2026): two projections ({Q, KV}), K = V = x·W_kv. Dense causal SDPA (GQA-general) + per-pass cached decode on the ONE-SLOT `LazyLatentCache` config, bit-exact to dense (sabotage-verified). Cache: one `[n_kv_heads·head_dim]` tensor/token/layer — 50% vs standard K/V at equal heads; ~98.4% vs MHA at MQA H=32/d=128 (the "98.5%" this row previously cited; the paper's 96.9% headline is a different config). Capability block, no checkpoint consumer yet; persistent-cache variant is a follow-up on the established D1 pattern. | `lazy_nn/two_proj_attention.rs` → this doc |
| **Symbolic-`k_len` flash `decompose`** | Absent (documented) | A `DynScalar`-length `Slice`/mask primitive (see the keystone). Today the symbolic decode oracle lives one layer up in the optimizer's `decode_flash` arm; `decompose` returns self by design. | Documented in [10-decisions-log 2026-07-03](architecture/10-decisions-log.md) (:404, :414-(2)) → also this doc |

---

## 3. Hyper-sparse Mixture-of-Experts & soft routing

**Built today:** six MoE models (Mixtral, Qwen2/3-MoE, DeepSeek-V2 @160 experts, Granite)
with a real router that even does in-graph top-K *selection* ([`fuel-core/src/lazy_nn/moe.rs`](../fuel-core/src/lazy_nn/moe.rs)). **But every one routes *densely*** — it computes all
N expert FFNs for every token and gates by the full softmax (~32× over-compute for a
256-expert/top-8 model, and not bit-exact to trained top-K). The genuine architectural
tension: data-dependent *values* work in a lazy DAG (router argmax/gather/scatter over
fixed shapes); data-dependent *shapes* (per-expert token counts) do not — the keystone.

| Gap | Status | What's needed | Home |
| --- | --- | --- | --- |
| **Sparse per-token expert dispatch** | Absent (dense today) | `Op::TopKRoute` (returns indices + weights + gated experts) + a gather-compute-scatter graph rewrite, riding on **data-determined dynamic shapes** (the keystone). Compute only *k* of N experts. | The *primitive* is planned via Phase 8.5 `Op::NonZeroIndices` + [data-dependent-shapes-design.md](session-prompts/data-dependent-shapes-design.md); the **MoE consumer is not called out** → this doc names MoE as a first consumer |
| **MoE load-balancing / aux-loss** | Absent | Auxiliary-loss, router z-loss, and/or aux-loss-free bias balancing to prevent expert stagnation. Training-side; belongs in `fuel-training`. | orphan → this doc |
| **Soft-MoE / dual-softmax** | Absent | Weighted linear token combinations ("soft patches") via dual-softmax. **Architecturally the friendliest** variant for Fuel — inherently dense, differentiable, *no* data-dependent shapes — yet no code. | orphan → this doc |

*Note:* the `Op::Branch` / "arms" machinery is **not** token routing — it is plan-time
selection among alternative *implementations* of the same math, decided by the optimizer
before execution (and currently inert scaffold). It cannot be repurposed for per-token
sparse dispatch. The retired eager `FusedMoe` grouped-GEMM (kernel-level sparse dispatch,
[`docs/moe-design-analysis.md`](moe-design-analysis.md)) was **dropped, not ported**,
during the lazy migration precisely because the lazy IR lacks the sparse-dispatch
primitive.

---

## 4. Test-time compute / inference scaling (MCTS, beam, self-consistency)

**Built today (strong substrate):** single-token autoregressive decode, full sampling
(greedy/temp/top-k/top-p/gumbel), a genuinely-wired **speculative decode** loop, the
**plan-once persistent decode** `DecodeSession` (~1.8×/token, proven concurrency-isolated
— N independent sessions from one shared `&model`), prefix-cache KV reuse, paged
attention, and `truncate_to` rollback. This is excellent raw material for tree/beam search.

**Architectural boundary (deliberate):** search *orchestration* (MCTS, beam, majority
voting) is **not Fuel's job** — it is a Rust-level realize loop in a *downstream*
consumer. Phase 9 keeps in-graph control flow out and provides theory-neutral hooks
instead, gated on a real consumer ("do not pre-build"). So the absence of MCTS/beam is a
*scope position*, not an oversight. What *is* under-captured are the two substrate pieces a
downstream search wrapper would need from Fuel:

| Gap | Status | What's needed | Home |
| --- | --- | --- | --- |
| **Batched multi-sequence decode** | Designed | Dynamic batch size + per-sequence lengths (ragged). The persistent decode graph is shape-keyed to `seq==1`, single sequence. Needed so M search hypotheses decode together. **2026-07-10 audit: the kernel substrate already exists and is unused** — baracuda alpha.72's `flash_decoding_{f16,bf16}` already takes a real batch dim over independent sequences (`blockIdx.z=batch`, per-sequence strides; one shared `k_len` scalar per call, lockstep decode only — a pinned kernel contract, not an oversight); `gemm_dense` already does weight-broadcast batching (used today for GQA); `FlashAttn`'s shape rule is already batch-generic. The gap is graph/session construction — fuel-core/fuel-graph never build a batch>1 call — not kernel research. | "recorded future" in [symbolic-extents design](session-prompts/symbolic-extents-and-persistent-decode.md) (:369-371) → this doc (make it a work item) |
| **Forkable / copy-on-write KV cache** | Absent | Cheap COW fork of a shared prefix into M divergent hypotheses. Today only `truncate_to` (rollback) + `cloned_persistent` (whole-map clone) exist. Paged attention gives the block substrate to build on. **2026-07-10 audit: confirmed there is no allocator to build on** — `Op::PagedAttn` (`fuel-core/src/lazy.rs:3253-3372`) is a bare compute-kernel signature (decomposes to `IndexSelect` gather + dense attention) with zero pool/refcounting infrastructure behind it — its one test hand-builds a trivial identity block table. The block-pool/COW subsystem (physical allocator, per-block refcounting, a real `BlockTable` type actually consumed by `KvCache`, refcount-aware eviction) is a from-scratch build comparable in scope to vLLM's own memory manager, not a wiring task. See the cross-branch-splicing entry below — its MVP recommendation sidesteps this allocator dependency entirely. | orphan → this doc |
| **Cross-branch KV content splicing** (new, related but distinct) | Designed (this audit) | Copy KV/residual content between concurrently-decoding, *persistent* branches — not a fork-to-one-winner search, ongoing "trains of thought" that stay alive and can cross-pollinate mid-generation at the orchestrator's choice. Full audit, source citations, and the reevaluation-cost menu below. | this doc (§ below) + memory `parallel-branch-kv-sharing-audit` / `multi-agent-serving-goal` |
| **Generation wrapper layer** | Absent (placeholder) | `fuel-inference::pipelines` is a literal empty `pub mod pipelines {}`; the decode loops live in `fuel-core` (below where the layer model puts them). A real wrapper ties model + sampler + KV + policy. | Layer model assigns it to `fuel-inference` (ROADMAP layer model); still unbuilt → this doc |
| **Search-on-generation orchestration (MCTS/beam/self-consistency)** | Absent (out of layer) | *By design a downstream consumer's job*, via Phase 9 hooks (`RuntimeHook`, persistent values). Recorded here so the boundary is explicit, not forgotten. | [Phase 9](../ROADMAP.md) (gated on a consumer) |

### Cross-branch KV content splicing (2026-07-10 audit)

A distinct proposal from the MCTS/beam fork-and-diverge pattern above: **N persistent,
independently-decoding branches of one loaded model that can explicitly copy KV (and
optionally residual) content from one branch's cache into another's, mid-generation, at
the orchestrator's choice** — not a search algorithm forking hypotheses that resolve to
one winner, but ongoing "trains of thought" that stay alive and can cross-pollinate.
Originates from a ChatGPT-drafted plan CireSnave brought for review; audited against
source across four conversation passes rather than taken on faith. Full reasoning trail
in the memory `parallel-branch-kv-sharing-audit` and `multi-agent-serving-goal` — this
entry is the durable, in-repo summary a future instance should start from.

**What was ruled out first.** `Op::Branch` is confirmed plan-time backend/kernel-variant
arm selection only ([`fuel-graph/src/lib.rs:1090`](../fuel-graph/src/lib.rs#L1090),
restated verbatim at `docs/frontier-paradigms-vision.md:169-171`) — no mechanism forks a
live sequence into independently-continuing streams, and it should stay that way; this
feature is a session/`fuel-core` concern, not a graph-IR control-flow change. Live
cross-branch *attention* (one branch's forward pass reading another's cache in-flight)
was also considered and ruled out — not what was actually wanted. See below.

**What's actually wanted: a splice, not a fork and not live attention.** Copy a K/V (and
optionally mid-stack residual) slice from source branch A's cache into destination
branch B's cache; B's own *unmodified* attention then reads it as ordinary history. This
sidesteps needing any new attention mechanism entirely.

**Confirmed substrate facts (source-verified, not assumed):**
- KV cache is batch=1 everywhere today. `KvCache` hardcodes leading shape dim 1
  ([`fuel-core/src/inference_context.rs:195`](../fuel-core/src/inference_context.rs#L195));
  `let batch = 1;` is a literal repeated across ~20 per-model `lazy_*.rs` files. No
  fork/branch/share/`Clone` method exists on `KvCache`/`DecodeSession`/`InferenceContext`.
- The batched-decode kernel substrate (see the row above) already exists and is unused —
  none of it is wired into fuel-core/fuel-graph's session path. The gap is graph/session
  construction, not kernel research.
- `Op::PagedAttn` IR node exists ([`fuel-core/src/lazy.rs:3253-3372`](../fuel-core/src/lazy.rs#L3253-L3372), real block-table batch semantics: `q:[B,Hq,Sq,D]`,
  `block_table:[B,max_blocks]`, `context_lens:[B]`) but is exercised by exactly one
  correctness test and never wired into `KvCache`/`DecodeSession`. **No allocator behind
  it — see the row above.**
- `WriteSlice`/`WriteSliceRotating`/`WriteSliceDoff` (the family an early draft plan
  proposed building branch-offsets on) are destructive-single-owner ops — only the
  writing op's own `NodeId` may read the result afterward
  ([`fuel-graph/src/lib.rs:859-861`](../fuel-graph/src/lib.rs#L859-L861)). Two branches
  destructively targeting shared storage hits an ordering conflict that
  `derive_ordering`/`insert_safety_copies`
  ([`fuel-graph/src/opt.rs:1531-1861`](../fuel-graph/src/opt.rs#L1531-L1861)) do **not**
  currently detect or reject — a silent-corruption risk if used naively for this. Not the
  right substrate.

**Recommended design: a host-level `KvCache` method, not a graph-routed one.** E.g.
`splice_from(&mut self, source: &KvCache, token_range, dest_offset)`, sibling to the
already-existing `truncate_to`/`cloned_persistent` escape hatches
([`inference_context.rs:376,629,1164`](../fuel-core/src/inference_context.rs#L376)) that
already reach into `KvCache` internals directly, outside the lazy graph. A real tensor
copy, not a pointer/shared-block graft — sidesteps the missing-allocator dependency
entirely and the `WriteSlice` multi-writer hazard (each session stays sole owner/writer
of its own buffer; the splice is a one-time copy-in, not ongoing shared ownership). Cost:
a real VRAM duplicate of the spliced slice plus a fast bandwidth-bound copy — cheap
relative to a forward pass, fine to do live between decode steps. `k_version`/
`v_version`/`AuthorityState` on `KvLayer` are confirmed-inert placeholders (checked
independently twice) — a host-level splice doesn't violate any currently-active tracked
invariant.

**Open integration question, not yet investigated:** whether `DecodeSession`'s
persistent/plan-once decode graph (shape/length-keyed validity) tolerates a
non-monotonic `cached_len` jump from a splice, or needs an explicit invalidate-and-rebuild
afterward.

**The reevaluation spectrum (why "foreign" KV might or might not be usable as-is).** A
spliced KV entry isn't a context-free fact — it's already a function of everything
earlier in the *donor's own* trajectory (baked in through however many layers of
attention mixing happened before the splice point). Whether that's useful signal or
harmful noise to the recipient is an open empirical question no one can predict from
architecture alone — and CireSnave is deliberately open to the "clash" between a donor's
foreign assumptions and the recipient's own being a *feature*, not just a risk to
engineer away. If it needs mitigating, there's a real graduated menu, cheapest first:

1. **Raw splice** — copy as-is. Zero new compute, position-foreign.
2. **+ RoPE delta-rotation** — RoPE rotations compose additively, so one delta-rotation
   `R(pos_B − pos_A)` converts a donor K vector into exactly what the recipient's own
   RoPE would have produced at the recipient's position — exact, no forward pass, needs
   one small new op. Fixes positional bookkeeping only, not contextual content. V is
   untouched (standard RoPE never rotates V).
3. **+ residual-stream continuation for the upper layers.** The deepest option, and one
   worth recording precisely because an earlier pass of this same audit stated it
   imprecisely: donating the donor's *residual stream* (not KV) at some layer L is a
   genuinely different lever than KV, because a residual — unlike KV — is a valid
   resumption point for *continuing the forward computation*. Mechanically: the recipient
   treats the donated tokens as new positions in its own sequence, seeding layer L+1's
   input with the donated residual instead of computing it the normal way (embedding →
   layers 1..L); from L+1 onward they're processed exactly like any newly-prefilled
   token — normed, projected to Q/K/V, attending over the recipient's own real cached K/V
   at that layer (this is where "combination" with the recipient's context actually
   happens, through ordinary attention — residuals across different token positions are
   never merged in a transformer) — producing genuinely recipient-contextualized K/V for
   layers L+1..N. **This does not replace needing the donor's raw K/V for layers 1..L.**
   The residual is a transient, single-pass bridge (exactly like any token's residual
   during ordinary inference — never cached across time steps in any standard
   transformer implementation); once consumed to seed the upper-layer computation, it's
   discarded. For the donated tokens to be durably attendable by the recipient's *own
   future* tokens at every layer (what a KV cache is for), layers 1..L still need real
   K/V populated — either the donor's raw K/V spliced in directly (rung 1/2, for the
   lower layers only) or there is nothing there to attend to at those layers at all. So
   this rung is an **enhancement layered on top of rung 1/2 for the upper portion of the
   stack, not a standalone cheaper alternative to KV splicing** — it trades real
   recipient-side compute (proportional to `(N−L)/N` of a forward pass over the donated
   tokens) for recipient-native upper-layer context. Byte accounting, corrected for
   needing both pieces: transmitting one layer's residual (size `d_model`) plus K/V for
   layers 1..L beats transmitting K/V for all N layers whenever you're skipping
   recomputation of more than roughly 2 layers' worth (worked example, Llama-3-8B-class
   config: `d_model=4096` ≈ 2 layers' worth of KV at `n_kv_heads=8, head_dim=128`) — a
   real net win when a meaningful chunk of the upper stack is being recomputed, a wash or
   worse for shallow (1-2 layer) skips. New capability needed: resume-forward-pass-from-
   an-arbitrary-layer given a supplied residual, instead of always starting at the
   embedding layer — well-defined, bounded, **not yet checked against how Fuel's
   `lazy_*.rs` per-model layer loop is structured** (a real next step if this rung is
   pursued).
4. **Seam/crossfade** — reprocess just the last few tokens of the graft (or synthetic
   bridge tokens) through the recipient's ordinary prefill against the already-spliced
   cache. Zero new infrastructure (this is already core functionality) — gives a locally
   recipient-native buffer right where new generation resumes; cost scales with the
   crossfade window, not the graft size. Orthogonal to and composable with rung 3 (a
   different axis — which *tokens* get full treatment, vs. which *layers*).
5. **Full re-prefill** — treat the source as plain text, re-prefill the whole graft
   through the recipient. The ceiling: 100% recipient-native, also zero new
   infrastructure (ordinary prefill), cost proportional to the full graft length.
   Requires carrying the donor's source *text* alongside every KV graft (near-free to do,
   and not optional — it's what makes rungs 4-5 possible at all).

**Auxiliary lever, cheap and orthogonal to all of the above:** a short textual marker or
preamble announcing "foreign context follows" at the splice point, run through
completely ordinary prefill. Invokes the model's trained-in handling of
quoted/reported speech and multi-document context rather than relying on raw spliced
activations to convey their own provenance on their own.

**What does *not* need any of this:** forwarding literal "intermediate math results" so a
branch can skip/guess computation has no separate lever at the tensor level — KV caching
already *is* transformers' only cross-position reuse mechanism (MLP has no cross-token
state to cache; the final hidden state has the same foreign-context problem as KV with no
offsetting benefit). But plain text/symbolic result-passing between branches ("branch A
already worked out X=Y, let B just use that") is a separate, well-precedented, standard
multi-agent-LLM pattern — cheap, robust, worth using independent of anything above.

**Sequencing relative to multi-agent serving.** CireSnave has confirmed multi-session /
multi-agent serving (running several concurrent agent sessions on shared hardware) as a
genuine near-term personal roadmap goal — get Fuel's basics working well first, then
multi-agent serving comes very soon after. This upgrades the block-pool allocator above
from *speculative* to *a real consumer, just not sequenced yet* — worth tracking, not
worth building before "basics" lands. It also means real multi-session serving will very
likely need actual session-lifecycle management for N concurrently-live KV caches anyway
(and plausibly, eventually, the allocator itself, for cheap shared-system-prompt reuse
across agents — the vLLM/SGLang production case) — building the splice feature's
implementation *after* that infrastructure exists is probably cheaper than building it
before, since it may get to piggyback on infra justified by a better, already-committed
reason. Recommended order: basics → multi-agent serving infra → revisit whether the
splice feature should upgrade from the plain-copy MVP to allocator-backed
pointer-sharing. None of that blocks getting an empirical signal on whether foreign-KV
grafting even produces something useful — rung 1 above needs zero new capability and
could run as a throwaway experiment at any time.

---

## 5. GRPO & verifiable post-training (RLVR)

**Built today:** a real training stack — SGD/AdamW, autodiff over the DAG (eager tape +
the Phase 7.5 symbolic graph-rewrite autograd scaffold), `cross_entropy` /
`fused_softmax_cross_entropy`, grad clipping, LR schedules, checkpointing, and a working
TinyLlama fine-tune. Training is **in scope**, pushed to the `fuel-training` leaf
([09-non-goals](architecture/09-non-goals.md): inference is the center of gravity, but
"the architecture doesn't reject training"). **GRPO/RLVR specifically are a clean blank.**

| Gap | Status | What's needed | Home |
| --- | --- | --- | --- |
| **GRPO** (Group Relative Policy Optimization) | Absent | Group sampling per prompt, relative-reward → advantage normalization, policy-gradient loss, optional KL-to-reference. **No separate critic** (that's GRPO's whole point). Greenfield on the existing `TrainState` + `backward` + AdamW + `cross_entropy` substrate — most naturally a new `fuel-rl` leaf or inside `fuel-training`. | orphan → this doc |
| **RLVR** (verifiable-reward harness) | Absent | A reward interface fed by code execution / math parsers giving binary rewards, wired into the GRPO loop. | orphan → this doc |
| **RNG / generator seam** | Absent (noted) | Where a `Generator` lives + how it threads through realize/autograd. Blocks group *sampling* as a graph op (and dropout, stochastic training). | Already flagged in [ROADMAP Deferred backlog](../ROADMAP.md) (the "RNG / generator seam" open design gap) |

*Note:* the only "RL" in the tree today is inherited Candle gym demos (DQN/DDPG — which
carry the critic GRPO exists to eliminate); unrelated to LLM post-training.

---

## Suggested sequencing (advisory — the ROADMAP owns the real order)

1. **Keystone first: finish data-determined dynamic shapes** (the data-dependent-shapes
   program). Unlocks MoE sparsity *and* underpins SSM/attention capacity buffers, and is
   already needed by Phase 8.5. Highest leverage.
2. **`Scan` `Op` (G3)** — ✅ **SHIPPED 2026-07-15** (`Op::Scan` Phase 1): a build-time
   enum extension; `selective_scan` + `ssd_chunk_scan` now decompose to it (G3 closed). SSM
   decode + GPU dispatch + hybrid models still follow.
3. **Attention compression** — KV-container generalization (structural) → MLA decode
   cache → weight-absorption / two-projection. Independent of 1–2 except for the shared
   symbolic-`k_len` slice.
4. **Training-side RL** — GRPO then RLVR, on the existing training stack; depends on the
   RNG/generator seam for sampling.
5. **Test-time compute** stays a *downstream* concern (Phase 9); Fuel's obligation is the
   substrate pieces (batched decode, forkable KV) — build those when a search consumer
   materializes.

Each item, when picked up, must still move at least one of the four
[identity-enforcement checks](architecture/01-identity.md#how-this-identity-is-enforced)
*more* true and none less — and lands lazy-only, test-gated, per the working agreement.
