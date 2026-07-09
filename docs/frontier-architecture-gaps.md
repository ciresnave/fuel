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
| **Higher-order `Scan` `Op`** (G3 basis gap) | Absent (documented) | A build-time `Op`-enum extension: an associative- or chunked-scan primitive. The `O(seqlen)` unroll is unbounded/un-re-fusable; the diagonal-SSM `CumSum` closed-form overflows for Mamba's `a<0`. `decompose` correctly returns self (never-crash surfaced gap). | Documented in [10-decisions-log 2026-07-03](architecture/10-decisions-log.md) (:406, :414) but **never scheduled** → orphan → this doc + ROADMAP backlog |
| **SSM autoregressive decode** | Absent | `Op::SelectiveScanWithInitState` (or a 6-input `selective_scan`) that consumes the prior step's `last_state`. Producer half exists (kernels already emit `last_state` as bundle slot 1); the consumer feedback is unbuilt. Blocks the Mamba/LFM2 decode loop — the actual KV-compression win. | orphan (source comment `lazy_mamba.rs:25-31`) → this doc |
| **GPU scan dispatch** | Absent | Wire the already-ported baracuda CUDA mamba kernels (`fuel-cuda-backend/src/baracuda/mamba.rs`) to `OpKind` dispatch (+ autograd). Today only `causal_conv1d` + `cumsum` dispatch on CUDA; the two scans do not. | orphan (source comment `mamba.rs:16-19`) → this doc |
| **GraniteMoEHybrid Mamba branch** | Partial (attention-only) | The Mamba branch currently bails; lands downstream of the `Scan` op + SSM decode above. | orphan (source comment `lazy_granitemoehybrid.rs:6-8`) → this doc |

*Sequencing:* the `Scan` op is the root; SSM decode + GPU dispatch follow; hybrid-model
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
| **Batched multi-sequence decode** | Designed | Dynamic batch size + per-sequence lengths (ragged). The persistent decode graph is shape-keyed to `seq==1`, single sequence. Needed so M search hypotheses decode together. | "recorded future" in [symbolic-extents design](session-prompts/symbolic-extents-and-persistent-decode.md) (:369-371) → this doc (make it a work item) |
| **Forkable / copy-on-write KV cache** | Absent | Cheap COW fork of a shared prefix into M divergent hypotheses. Today only `truncate_to` (rollback) + `cloned_persistent` (whole-map clone) exist. Paged attention gives the block substrate to build on. | orphan → this doc |
| **Generation wrapper layer** | Absent (placeholder) | `fuel-inference::pipelines` is a literal empty `pub mod pipelines {}`; the decode loops live in `fuel-core` (below where the layer model puts them). A real wrapper ties model + sampler + KV + policy. | Layer model assigns it to `fuel-inference` (ROADMAP layer model); still unbuilt → this doc |
| **Search-on-generation orchestration (MCTS/beam/self-consistency)** | Absent (out of layer) | *By design a downstream consumer's job*, via Phase 9 hooks (`RuntimeHook`, persistent values). Recorded here so the boundary is explicit, not forgotten. | [Phase 9](../ROADMAP.md) (gated on a consumer) |

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
2. **`Scan` `Op` (G3)** — an independent build-time enum extension; unblocks
   `selective_scan` + `ssd_chunk_scan`, then SSM decode + GPU dispatch, then hybrid models.
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
