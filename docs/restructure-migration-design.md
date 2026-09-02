# Fuel restructure — migration design

**Status**: Stage 1 **SHIPPED** 2026-09-02 (`a2027651`, PR #35); Stages 2–4 not yet executed. Originally drafted 2026-08-26, measured at `1b6ae698` (pre-Stage-1).
**Reviewers requested**: Vulkane (live external backend), Lightbulb (external consumer of `fuel-core`), Unpopped (provider, if the FKC surface publishes), the portfolio PM (claims-against-refs).
**Author**: Fuel architect. Corrected four times before landing; see §10.

---

## 0. What this document is, and what it is not

**It is not a redesign.** The destination is already ratified in
[`docs/architecture/02-layers.md`](architecture/02-layers.md) v0.5 — Foundation
boundaries, the Models tier, the format/interchange tiers, the stopping rule for
new crates, and the `fuel-core` dissolution trajectory. That document is the
constitution and this one does not amend it.

**It is the path.** 02-layers pins *where things go*; nothing until now pinned
*how we get there from a 184,000-line `fuel-core`*, in what order, what breaks at
each step, and what each external consumer has to do. Two lanes are stopped
waiting on exactly that.

Where this document proposes something 02-layers does not already say, it is
marked **NEW** and carries its own justification against the stopping rule.

---

## 1. What is already decided, what has landed, what is new

### Ratified in 02-layers v0.5 (not re-opened here)

- Foundation crate boundaries: `fuel-ir`, `fuel-graph`, `fuel-memory`,
  `fuel-dispatch`, `fuel-hardware`, `fuel-backend-contract`, plus post-fission
  `fuel-tensor` + `fuel-autograd`.
- Models tier: `fuel-model-core` (the `Model` trait, the registry,
  `AutoModel::from_path`, link-time distributed registration) + `fuel-model-*`
  leaves, with `fuel-transformers` as an optional umbrella.
- Generic building blocks (RoPE, RMSNorm, GQA attention, SwiGLU MLP) stay in
  **`fuel-nn`**, not in per-family crates.
- **The stopping rule**: a new crate is justified only when a class of consumer
  uses one side and not the other; *the split must be in the dependency graph,
  not just the file layout*; speculative splits are rejected.
- **The reason `fuel-core` must go at all**: the name collides with an existing
  crates.io crate. This is a publish blocker, not an aesthetic preference.
- The stopping rule applied to models: **high-demand architectures split now,
  the long tail extracted lazily.** Big-banging every model into its own crate
  is explicitly rejected.

### Landed (B0.1–B0.5)

`fuel-core-types` → **`fuel-ir`** · hardware discovery → **`fuel-hardware`** ·
`fuel-storage` → **`fuel-memory`** · backend-contract traits →
**`fuel-backend-contract`**.

### Not landed — and it is the whole remaining job

**`fuel-core` itself has not dissolved.** 02-layers describes it as "an umbrella
facade [that] dissolves into the crates it re-exports." It *is* a facade — 192
`pub use` / `pub mod` lines in `lib.rs` — but it is a facade that also **contains
184,432 lines of implementation**.

### NEW, from CireSnave (2026-08-26)

1. **`fuel-model`** — a *model-authoring surface*: a text language a model author
   writes to create a model. **See §4: this name collides with the ratified
   `fuel-model-core` / `fuel-model-*` tier, which means something else.**
2. **`fuel` becomes a real facade crate** exporting the public surface.
   **See §3 — this is the migration's central lever, and it is nearly free.**
3. **`fuel-error`** as a shared crate every other crate needs, open to being
   merged with other genuine must-haves.

---

## 2. Measured current state

All figures at `1b6ae698`, counted rather than recalled.

```
36 workspace members, 26 default-members

fuel-core      184,432 lines / 220 files      <- the crate that must dissolve
fuel-dispatch  138,772 lines / 125 files
fuel-graph      48,410 lines /  32 files
fuel-ir         15,755 lines /  36 files
```

### `fuel-core` is not 184k lines of framework

```
134,800  (73%)  model definitions — 147 `lazy_*` files
 25,515  (14%)  lazy.rs — the user-facing tensor API
  8,825   (5%)  runtime bridge + the Judge
  8,111   (4%)  serving / decode (kv_block_pool, persistent_decode, inference_context)
  5,548   (3%)  everything else (31 files: nf4, device, decode_shape, telemetry, …)
  1,633   (1%)  training
```

**Three-quarters of the "dissolving crate" is a model zoo, not framework.** That
single fact does more to make this tractable than any design decision below.

### The blast radius is six crates, not thirty-six

In-tree dependents of `fuel-core`: `fuel-datasets`, `fuel-examples`, `fuel-nn`,
`fuel-onnx`, `fuel-tensor-tools`, `fuel-transformers`. Plus **Lightbulb**,
externally.

### Corrections to CLAUDE.md found while measuring

Three claims in the working agreement are false at head. Recorded here because
the working agreement is loaded into every session:

| CLAUDE.md says | Measured at `1b6ae698` |
|---|---|
| `fuel-nn` — "**Never existed.** The NN surface lives in `fuel-core`" | **Exists**: 23 files, 8,859 lines, a workspace member, and 02-layers assigns it the generic building blocks |
| `fuel-tensor-tools` "was **archived** 2026-08-01" | **Exists**: 626 lines, still in `members` |
| the `fuel-wasm-examples/*` glob "is present in both `members` and `default-members`" | **Absent**: `grep -c wasm Cargo.toml` → 0 |

Per the working agreement's own rule — *when one entry in a list you wrote is
corrected, re-measure the whole list* — the crate-name glossary needs a pass.

---

## 3. The central lever: `fuel` is already the name everyone imports

```
Root manifest:  fuel = { path = "./fuel-core", package = "fuel-core" }

Consumer imports, excluding fuel-core itself:
    fuel::         1018
    fuel_core::      18      (98.3% / 1.7%)

  fuel-examples  553 : 2      fuel-nn  219 : 0      fuel-onnx  21 : 0
  fuel-datasets   13 : 0   fuel-transformers 3 : 0  fuel-tensor-tools 2 : 1
```

**Nobody depends on the name `fuel-core`. They depend on the *paths* under
`fuel::`.**

So the dissolution does not have to be a breaking change:

> **Promote `fuel` from a manifest alias to a real facade crate that re-exports
> the same paths from the fissioned crates. `fuel-core` then empties out
> underneath it, and no consumer edits an import.**

This turns the riskiest-looking part of the restructure into the safest, and it
gives a **mechanical, checkable gate**: the facade's public path set must be
identical before and after each move. That is diffable — `cargo public-api` or an
equivalent path dump — and it is the single gate that makes every later stage
safe to attempt.

**The 18 `fuel_core::` sites are a one-line sed and should be converted first**,
so the alias is the only route in before anything moves.

---

## 4. ⚠️ A name collision to resolve before either name is fixed

CireSnave approved **`fuel-model`** for *"a text language that model authors can
enter as input to create a model."*

02-layers v0.5 already specifies **`fuel-model-core`** and **`fuel-model-*`** —
the `Model` trait, the architecture registry, `AutoModel::from_path`, and one
crate per model architecture. **That is a different thing**: the *definitions and
registry*, not an authoring language.

`fuel-model` and `fuel-model-core` sitting side by side, meaning
"authoring surface" and "model registry", is the kind of thing that reads fine in
a design document and is unusable in an import list.

**Options, cheapest first:**

| | Authoring surface | Registry + definitions |
|---|---|---|
| **A** | `fuel-author` | `fuel-model-core` / `fuel-model-*` *(unchanged, ratified)* |
| **B** | `fuel-model` | `fuel-arch-core` / `fuel-arch-*` *(renames a ratified tier)* |
| **C** | `fuel-model-lang` | `fuel-model-core` / `fuel-model-*` *(both keep the prefix; still adjacent)* |

**Recommendation: A.** It leaves a ratified tier alone, and *author* names the
act rather than the artifact — which is what distinguishes the two things.
**This is CireSnave's call and it should be made before either name is written
into a manifest**, because a crate name is the one decision that is expensive to
revisit after publish.

**Registry checked 2026-08-26** (raised by the portfolio PM, re-verified here
against `index.crates.io` with HTTP status codes and a negative control —
`fuel-nonexistent-xyzzy-control` returns 404, the same as the free names, so the
instrument discriminates in both directions):

```
TAKEN   fuel         0.1.0,    1 version,  live     <- see the warning below
TAKEN   fuel-core    0.48.2, 125 versions, live     <- the unrelated blockchain client
FREE    fuel-author · fuel-model · fuel-model-core · fuel-tensor · fuel-error
```

**Every candidate name in §4 is free, so availability does not decide the
collision — the argument there is purely semantic.**

### ⚠️ But `fuel` itself is taken, and §3's lever is named after it

`fuel 0.1.0`, one version, no successor, unyanked — the classic squat shape, and
crates.io does not reclaim those on request.

**This does not break the lever; it splits it in two, and the distinction is the
PM's:**

- **In-tree, the lever is untouched.** `fuel = { path = "./fuel-core", package =
  "fuel-core" }` is a *local rename*; the registry is irrelevant to it. Every
  claim in §3 about imports not changing holds exactly as measured. Lightbulb
  consumes Fuel **by git rev**, not from the registry, so their guarantee holds
  too.
- **At publish, the facade cannot be called `fuel` on crates.io.** The published
  name and the path root can differ, and **the consumer controls the path root
  themselves**: `fuel = { package = "<published-name>", version = "…" }` in their
  manifest keeps every `use fuel::…` working. So this costs a name, not the
  design.

**And note what the 1018-vs-18 measurement does and does not license.** It is a
fact about **paths**. `fuel` as a *registry name* is a different axis, and the
ratio speaks to neither its availability nor its publishability. **A true
measurement attached to a wider claim than it supports** — the defect this
document's own §6 is full of, arriving in its headline argument. The lever
survives; the sentence needed splitting.

---

## 5. The ordering

Four stages. **Each is independently valuable and independently revertible**, and
no stage begins before its predecessor's gate is green.

### Stage 1 — Make `fuel` real (small, unblocks everything)

**SHIPPED 2026-09-02 — `a2027651`, PR #35.** The facade crate landed: a
byte-identical public surface (`pub use fuel_core::*` + an explicit `bail`
re-export) with a 1:1 feature-forwarding gate, consumer impact nil (all 6
`fuel::` consumers compiled unchanged). The `fuel_core::` doc/prose sites are a
separate follow-up (PR #43) — 7 converted, 7 left after a STALE/HISTORICAL/PINNED
read, 2 deferred to Stage 2 (below). The original plan text is kept for the record:

Convert the 18 `fuel_core::` sites to `fuel::`. Create `fuel` as a real crate
whose `lib.rs` re-exports exactly what `fuel-core/src/lib.rs` exports today.
`fuel-core` becomes its only dependency.

**Gate**: the public path set is byte-identical before and after.
**Consumer impact**: none.

### Stage 2 — Move the model zoo out (73% of the problem, mechanical)

Move the 147 `lazy_*` files into the Models tier. **As a group, behind the
existing `fuel-transformers` umbrella — not into 147 crates.** 02-layers'
stopping rule rejects that big-bang explicitly; leaves get extracted lazily when a
real single-model consumer appears.

`fuel-core` goes from 184k to ~50k lines. These files are leaf consumers of the
tensor API; they are moved, not rewritten.

**Gate**: facade path set unchanged; `fuel-examples` (553 `fuel::` imports, the
heaviest consumer) compiles untouched.

**Deferred doc-pointers to fold in here.** Two example source-pointers were left
as `fuel_core::` in the Stage 1 facade-rename follow-up because Stage 2 relocates
the module they name: `fuel-examples/examples/debertav2/main.rs:347` (a `bail!`
string naming `fuel_core::lazy_debertav2`) and `.../xlm-roberta/main.rs:142` (a
comment naming `fuel_core::lazy_xlm_roberta`). Both spellings — `fuel_core::` and
the canonical `fuel::` — go stale once these modules land in `fuel-transformers`,
so converting earlier was churn for no reader. **They are RUNTIME text (a `bail!`
string and a comment), not `[intra-doc links]`, so no rustdoc gate flags them if
this stage forgets** — retarget them to wherever the module lands as part of this move.

**And this is the first real test of the §4 authoring surface**: whatever
`fuel-author` ends up being, it has to be able to express what already exists in
`lazy_flux.rs` and `lazy_mmdit.rs`. **147 worked examples is a better spec than
any greenfield design**, and if the surface cannot express them, the surface is
wrong.

#### EXECUTED 2026-09-02 (commits `fe765b38` repoint + `798e99cf` move)

**147 candidates → 146 moved, 1 stays (`147 − 1`).** Do not restate the moved count
as 147; ROADMAP and this recording agree on 146 with `lazy_latent_cache` named.

- **Forced carve-out — stay-list = {`lazy_latent_cache`}.** The criterion is
  MECHANICAL, not "is it a model" or "is it a cache": a `lazy_*` module STAYS in
  `fuel-core` iff any crate **at or below the Models tier** references it in code —
  moving it would create an upward edge and cycle. (GAP-265: cargo enforces
  acyclicity, not layer DIRECTION, so this is classified by hand, not by "it
  compiles.") Sole forcer: `fuel-nn/src/modules/two_proj_attention.rs` uses
  `lazy_latent_cache::LatentCache` in production; transitive closure empty.
  ⚠️ **`lazy_kv_cache` MOVES** — it has no at/below-Models code consumer, so the
  mechanical criterion keeps one cache module, NOT the cache family. This diverges
  from the "keep the caches together" gloss deliberately; reuniting `lazy_kv_cache`
  with `lazy_latent_cache` is a Stage 3 question.
- **The cycle it resolves.** `fuel-transformers` depended on the `fuel` FACADE;
  re-exporting its models from the facade would cycle `facade → fuel-transformers →
  facade`. Fix: repoint `fuel-transformers → fuel-core` (Foundation). The facade
  re-exports `fuel_transformers::models::*`, so `fuel::lazy_bert` still resolves —
  path set unchanged.
- **Gate met.** `fuel-examples` (the heaviest consumer) compiles UNTOUCHED; its
  `use fuel::lazy_X` imports resolve through the facade re-export unchanged.
- **Reference census, re-run AFTER the rebase** over `fuel-transformers/src/models/`
  (a corpus measurement: a clean rebase attests to mergeability, not that the corpus
  still matches): `fuel_core::` 2354 (524 `use` imports) · `crate::models::` 221 ·
  dangling `crate::` not `models::` **0** · code `fuel::` facade **0** (the single
  `fuel::` hit is `models/mod.rs` header PROSE describing facade resolution) ·
  `fuel_transformers::` 24. `cargo check -p fuel-transformers -p fuel --locked
  --all-targets` exits 0, so the corpus resolves and `Cargo.lock` is consistent.
- **Deferred doc-pointers folded in** (the two this section named):
  `debertav2/main.rs:347` (a `bail!` string) and `xlm-roberta/main.rs:142` (a
  comment) retargeted `fuel_core::lazy_X` → `fuel::lazy_X`; both are runtime text no
  rustdoc gate flags. ⚠️ The plan's claim above that "the canonical `fuel::` also
  goes stale" is SUPERSEDED and was false: the facade re-exports
  `fuel_transformers::models::*`, so `fuel::lazy_debertav2` resolves — `fuel::` is
  the correct stable target.
- **Doc gates.** `cargo test --doc -p fuel-transformers` compiles **6** executable
  doctests (all pass); the gate was born-red-confirmed by sabotaging a FENCED
  `fuel_core::Error` ref in `lazy_bert.rs` → `E0425` fail, restored → pass
  (recompiled, 4.91s vs 0.48s). Reference dispositions beyond code: **33** intra-doc
  `[...]` links (`cargo doc` `broken_intra_doc_links`) + **29** bare `` `code
  spans` `` (NO gate sees them; read individually — 26 pre-existing + 3 new header).
  The aspirational subset (**48** = 33 links + 15 bare) to an unbuilt
  `models::{llm,audio,multimodal}::` / `_models_retired::` taxonomy is PRE-EXISTING,
  not introduced by the move, and tracked as **GAP-266**.
- **Visibility — 7 changes, all `fuel-core`-owned:** 4 fns
  (`refresh_decode_session`, `offer_flash_decode_arm_for_region` in `lazy.rs`;
  `compute_decode_token_host`, `upload_decode_token_data` in `persistent_decode.rs`)
  + 1 method (`run_backbone_with_rope_tables`, E0624) + 2 types (`DecodeTokenHost`,
  `SessionDisposition`). The two types are `pub` as a Stage 2 CONSEQUENCE — a
  `pub fn` cannot return a private type and the models thread them by inference — NOT
  an API judgement; **revisit when Stage 3 settles the serving/decode destination.**
  `Tensor::inner` stays `pub(crate)` (GAP-264): the zoo's 24 uses were rewritten to
  the existing `graph()`/`node_id()` accessors, and the crate boundary now enforces it.

### Stage 3 — Fission the ~50k remainder

Now a normal-sized refactor, and only now are the boundaries visible:

| Content | Destination | Note |
|---|---|---|
| `lazy.rs` (25.5k) | **`fuel-tensor`** | ratified; Lightbulb is the named consumer |
| runtime bridge (`pipelined_bridge`, `factories`, `scheduling`) | **`fuel-dispatch`** | joins the ranker/executor already there |
| the **Judge** (4.8k) | **RULED — split. See §5.1.** | runner is a leaf tool; the oracle is not |
| serving / decode (8.1k) | **open** | plausibly `fuel-inference` (exists) |
| `train.rs` | **`fuel-training`** (exists) | |
| the rest (5.5k, 31 files) | distribute | `nf4`→`fuel-quantized`, `device`→`fuel-hardware`, … |

### Per-row membership + pre-flight (measured 2026-09-02, at `main` `131e8b84`)

⚠️ **The rows above cannot be trusted row-by-row.** Each names a *size* and a
*destination* and never a *file set* — and a size cannot disagree with anything, so a
multi-layer row looks exactly like a single-layer one. The `serving/decode` hole was
filed as ONE row for exactly this reason. This pre-flight was required to make the rows
falsifiable. Result: **three of six rows conflate layers, "the rest" spans four tiers,
and two real dependency cycles were caught before execution.**

Pre-flight test per row: does any crate *at or below* the destination consume the code
(consumer direction), and can the destination reach what the code depends on (dependency
direction)? Either failing is a cycle — the defect Stage 2 hit (GAP-265: cargo enforces
acyclicity, not layer direction).

⚠️ **None of the 45 `fuel-core/src` files carries a `//! **Layer**:` declaration.** Six
crates in this workspace self-declare their layer; none of these files does. **Every
placement below was inferred from the dependency graph, not read off** — a future reader
must not mistake these for self-declared.

Layer order (bottom→top): `fuel-ir`/`fuel-graph`/`fuel-memory`/backends → `fuel-dispatch`
→ `fuel-core` (Foundation/IO) → `fuel-nn` (NN) → `fuel-transformers` (Models) →
`fuel-inference`/`fuel-training` (top leaves).

#### Row 1 — `lazy.rs` (25,925) → `fuel-tensor` · ⚠️ MULTI-LAYER (≥3 destinations)

| Member (by concern) | Destination | Consumers | Pre-flight |
|---|---|---|---|
| Tensor API (bulk) | `fuel-tensor` (Foundation) | all tiers above | CLEAN |
| `LlamaModel`/`PhiModel` (`impl` at :8103, pre-trait models) | `fuel-transformers` (Models) | `fuel-inference` + intra-Models | architectures, not tensor API |
| Paged serving methods (:8674+ `forward_paged_step`, …; take `&mut DeviceKvPool`/`SessionHandle`) | `fuel-inference` (serving, with Group B) | **only** `fuel-inference` | ⚠️ must NOT ride into `fuel-tensor` — `fuel-tensor→fuel-inference` cycle |

The single "→ `fuel-tensor`" destination is wrong for two of the three concerns.

#### Row 2 — runtime bridge (`pipelined_bridge` 2680, `scheduling` 1481, `factories` 331) → `fuel-dispatch` · 2 clean, 1 mis-filed

| Member | Destination | Pre-flight |
|---|---|---|
| `pipelined_bridge.rs` | `fuel-dispatch` | CLEAN dep-wise (code uses `fuel_ir`/`fuel_graph`/`fuel_memory`); its 7 `fuel-core` refs are **doc-links only**, but they point at `crate::inference_context`/`lazy::Tensor` and **break on the move → must repath to Group A's home** |
| `scheduling.rs` | `fuel-dispatch` | CLEAN (0 `fuel-core` code refs) |
| `factories.rs` | ⚠️ **`fuel-judge`, not `fuel-dispatch`** (RULED 2026-09-02, OPT A) | **mis-GROUPING corrected — it was never executor code.** `factories.rs:41 use crate::lazy::Tensor` is real code and `fuel-dispatch` is *below* `fuel-tensor` → cycle. Its **sole code consumer** is `judge/mod.rs` (the §5.1 runner) — it is the Judge's realize *facade* — so it belongs in `fuel-judge` (a top leaf, so `Tensor`/`Device`/`pipelined_bridge`/`StorageCache` are all downward, CLEAN, zero decoupling). |

⚠️ **A stale prose consumer list is why this row looked ambiguous — a finding in its own
right.** `factories.rs`'s module doc (line 7) names "the `crate::probe` enumerator" as a
consumer; but `crate::probe` is `pub use fuel_hardware::probe`, and `factories.rs:34-36` —
twenty-seven lines below — records that device enumeration moved to `fuel-hardware`'s
`HardwareEnumerator` in B0.2. **The file contradicts itself: a doc naming a consumer that
no longer exists, sitting above the note that explains why it does not.** A prose consumer
list is a *cache* — nothing re-derives it, so it goes stale silently. It is the same
defect as the Stage 3 rows themselves (an unfalsifiable claim nothing re-checks), and a
better example, because here the refutation was already in the file. **Two doc-link
repaths also ride this move:** `pipelined_bridge.rs:316` and `:1233` reference
`crate::factories::{BridgeRealizer, Realizer}` in doc comments, and since `factories` →
`fuel-judge` while `pipelined_bridge` → `fuel-dispatch`, those links become cross-crate in
both directions at once.

#### Row 3 — the Judge (`judge/mod.rs` 3910 → `fuel-judge`; `judge/oracle.rs` 439 + `judge/cache.rs` 412 → `fuel-dispatch`) · RULED (§5.1), premise VERIFIED

Pre-flight confirmed 2026-09-02: `oracle.rs`/`cache.rs` non-test imports are only
`fuel_dispatch` + `fuel_ir` — **zero** `fuel-core`-Foundation code deps, so the
`→ fuel-dispatch` half carries no cycle (the `factories` cycle does not recur here).
Execution note: their tests reference `crate::judge::{test_equiv_key,
PROFILE_REPORT_VERSION}` from `judge/mod.rs` → `fuel-judge`; those helpers must
move/repath on the split — test wiring, not a layering fault. `factories.rs` joins this
crate's runner (see Row 2).

#### Row 4 — serving / decode (8,926) → SPLIT

| Group | Members | Destination | Pre-flight |
|---|---|---|---|
| A — decode-graph machinery | `persistent_decode` 1482, `inference_context` 2671, `decode_shape` 471 | **new `fuel-decode`** (Foundation/NN level) | consumed by `fuel-transformers` (Models) + `fuel-inference`; **`fuel-inference` FAILS** (Models is below it). Crate forced by **L880** (Models must hold no sessions; A holds `DecodeSession`/`KvCache`/`InferenceContext`). Stopping-rule class *Models-without-A* is real (7 of 146 models decode) but its benefit is conditional on a `decode` feature gate over the 7 families. |
| B — serving orchestration | `kv_block_pool` 1720, `kv_block_pool_device` 2238, `decode_state_spec` 344 | **`fuel-inference`** | consumed only by `fuel-inference`; PASSES. `multi_session.rs` (4163) already there. `lazy.rs` paged methods peel here (Row 1). |

Verified A has **zero** code dep on B (only doc-links), so B is free to go up.

#### Row 5 — `train.rs` (1,633) → `fuel-training` · CLEAN

Only consumer is `lib.rs`'s `pub mod train;` (moves with it); no below-consumer;
`fuel-training` is a top leaf. PASSES.

#### Row 6 — "the rest" (~25 files, ~5.5k) → distribute · ⚠️ FOUR tiers, not one

| Tier | Files |
|---|---|
| Foundation (`fuel-tensor`/`fuel-ir`) | `dtype` 111, `shape` 52, `error` 26, `storage` 19, `layout` 5, `strided_index` 5, `planner` 117, `model_progress` 138 |
| Kernels (`fuel-quantized`) | `nf4` 698, `quantized/{arch,gguf_mmap,gguf_file,imatrix_file,mod}` ~449, `utils` 111 |
| Backends | `accelerate` 477, `mkl` 419 → CPU backend crates; `cuda_backend`/`vulkan_backend`/`metal_backend`/`cpu_backend`/`backend`/`dyn_backend` → re-export shims |
| `fuel-dispatch` | `telemetry` 364 |
| IO | `safetensors` 241, `hf_config` 111 → `fuel-core` IO remnant / `fuel-io`; ⚠️ **`quantized/tokenizer.rs` 333 is tokenizer glue → IO, not Kernels** |
| test-support | `test_utils` 124 |

The named examples (`nf4`→`fuel-quantized`, `device`→`fuel-hardware`) are correct but
partial. All destinations sit at/below their consumers, so each passes the pre-flight;
the finding is that the row is four rows.

#### Carve-out — `lazy_latent_cache.rs` (398) → `fuel-nn`

Lowest real consumer is `fuel-nn` (`two_proj_attention.rs`); also `fuel-transformers` +
`fuel-inference`, both above. Not `fuel-transformers` (that is the Stage 2 cycle). Kept
apart from `lazy_kv_cache` — which has no Models-only deps, so reunion is *feasible* — but
feasibility is not a reason; the separation stands with this recorded rationale, and the
burden to reunite is a positive argument, not tidiness.

#### What the pre-flight caught before execution

- `lazy.rs` paged methods → `fuel-inference` would have cycled `fuel-tensor→fuel-inference`
  had `lazy.rs` moved monolithically to `fuel-tensor`.
- `factories.rs`'s `use crate::lazy::Tensor` → `fuel-dispatch` would have cycled
  `fuel-dispatch→fuel-tensor`.

Both are the GAP-265 defect (an upward edge that compiles until the crate boundary
exists), and both were invisible in a row that named only a size and a destination.

### 5.1 The Judge splits, and the line is already drawn in its file layout

**Ruled 2026-08-27.** This was left open in the first draft, then blocked on a
premise §6.3 has since retracted. Settling it rather than leaving it hanging.

**Measured shape:**

```
fuel-core/src/judge/mod.rs      3910   the RUNNER  — builds graphs, realizes,
                                                    times, catch_unwinds per cell
fuel-core/src/judge/oracle.rs    439   CONSUMPTION — ProfileReport -> JudgeOracle
fuel-core/src/judge/cache.rs     412   CONSUMPTION — process-wide DispatchTable cache
```

**The stopping rule is satisfied, one-directionally, which is enough:**

- **Ranker without runner is a real class, and it is the majority.** Every
  inference consumer reads a profile and never produces one; with no profile the
  ranker falls back to Layer-1 static costs and works. **Lightbulb is exactly
  this consumer.** Today they would link **3,910 lines of profiler they never
  execute.**
- **Runner without ranker is not a class.** The Judge's output has one consumer.

So the split is **not** symmetric and does not need to be. The test is *"a class
of consumer that uses one side and not the other"* — one such class is sufficient.

**Ruling:**

| | destination | why |
|---|---|---|
| `judge/mod.rs` — the runner | **`fuel-judge`** (new, leaf) | nothing depends on it; it belongs in 02-layers' *Use-Case Orchestration* tier beside `fuel-inference` and `fuel-training`, which that tier already describes as *"leaf crates — nothing depends on either"* |
| `judge/oracle.rs` + `judge/cache.rs` | **`fuel-dispatch`** | they sit next to the ranker that consumes them; `pipelined_bridge` already calls `cached_oracle()` on the realize path |

**The schema is already where it belongs and does not move:** `ProfileEntry`,
`ProfileReport`, `DispatchTable` and `PROFILE_REPORT_VERSION` are in **`fuel-ir`**
— the crate the restructure keeps. Producer, consumer and schema were always three
things; only the first two were in one crate.

⚠️ **One coupling must invert, and it is the whole cost of this split.**
`cache.rs:160` calls `Judge::default().run(&probe)` — the *consumption* side
invoking the *runner* to auto-populate on a cache miss. **That single call is what
would drag the profiler into every consumer's binary.**

**Fix:** `populate_dispatch_table()` takes a report (or a producer trait) rather
than constructing a `Judge`. **Convenience coupling, not structural** — the
auto-populate behaviour survives, it just gets injected instead of hard-wired,
and `fuel-judge` becomes the thing that injects it.

**Name.** `fuel-judge` — free on the registry (measured, with a negative
control), and it matches every existing doc comment, which all say "the Judge".
**Note this name leaked once as exploratory and a downstream project built a live
trigger on it** (§10); settling it deliberately is what makes reusing it safe
rather than confusing.

**No `fuel-autograd` split is proposed here.** 02-layers names it, but the
stopping rule requires a consumer that wants one side and not the other, and that
pressure should be demonstrated at the time rather than assumed now.

### Stage 4 — Retire `fuel-core`

The crate is empty. Delete it, drop the manifest alias, publish under `fuel`.

**`fuel-error` (NEW)** lands wherever it is first genuinely needed by two crates
that do not otherwise depend on each other — **not up front.** The stopping rule
applies to it exactly as to everything else, and "every crate needs it" is a
prediction until two crates need it and cannot share a nearer home.

---

## 6. Where does a trait go when its crate dissolves?

Three instances arrived independently, which is why this is a section and not
three cases.

### 6.1 `Realizer` — the internal instance

`pub trait Realizer` lives in `fuel-core/src/factories.rs:53`. Two implementors
(`BridgeRealizer`, plus one test stub), **zero external implementors across all
12 portfolio repos**, and it is not named outside `fuel-core`.

**It follows its implementor into `fuel-dispatch` at Stage 3.** `pub` here is a
visibility keyword, not a distribution fact, so this is a move and not a
publication event.

**Live interaction with ROADMAP item 7**: that work is adding
`last_kernel_revision()` to this trait *now*. The lane has been told explicitly
not to let the restructure change what they build — a lane optimising for a design
that does not exist yet would be guessing, and the guess would land in code the
design then has to accommodate.

### 6.2 Lightbulb's `impl From<CudaDevice> for Device` — the external instance

**Lightbulb's dependency on `fuel-core` is a trait impl, not a function call.**
Their port question is *"where does that impl live in the new shape?"*, and the
design must **answer** it rather than leave it derivable.

**Answer**: `Device` moves to `fuel-hardware` (`device.rs`, 493 lines, Stage 3);
`CudaDevice` is `fuel-cuda-backend`'s. The impl is a cross-crate bridge, so it
lives with whichever side is closer to being the orphan — **`fuel-cuda-backend`**,
which already depends on the hardware layer.

**Under §3's lever Lightbulb's import does not change at all** if they reach it
through `fuel::`. If they name `fuel_core::` directly, they are in the 1.7% and
we owe them a one-line change and advance notice. **They should confirm which,
against their own tree, rather than us inferring it.**

### 6.3 `fkc::verify`'s two device roles — the provider instance

Vulkane measured the seam at `7954f73c`: `KernelInvoker` and `HostTensor` are
already backend-neutral; the coupling is `&CudaDevice` at the entry points, plus
`fkc::verify` being private.

⚠️ **RETRACTED 2026-08-27, BY THE PARTY WHO SUPPLIED IT.** This paragraph
previously ended *"`Storage` is a separate, larger coupling that must not be
conflated with this one."* **Vulkane withdrew that sentence in both directions at
once**, and the corrected version is more useful than the original:

- **`Storage` IS on the provider critical path.** `CudaInvoker::invoke` does not
  merely take host bytes and return host bytes — it **constructs
  `fuel_memory::Storage`** to call the kernel, and *any* `KernelInvoker` must,
  because `BindingEntry.kernel` is a `KernelRef` over `Arc<RwLock<Storage>>`.
  **The trait's SIGNATURE is host-only; its CONTRACT is "run this entry", and
  running one touches `Storage`.** In their words: *"I read the signature and
  stopped."*
- **And it does not matter, because `BackendStorage::Vulkan` already exists**
  (`fuel-memory/src/lib.rs:73`, behind the `vulkan` feature). So the correct
  sentence is **"on the path, and already satisfied"**, not "separate and
  larger". The decision to leave `Storage` alone stands; the stated reason for it
  was wrong.

⚠️ **AND A SECOND CORRECTION, THIS ONE OF THE ARCHITECT'S.** I told Vulkane that
a trait with one implementor is a shape drawn around existing code, they
generalised it back, and **the premise was false for this trait.**
`KernelInvoker` has **ten implementors** — `CpuInvoker`, `CudaInvoker`,
`VulkanInvoker`, `ExactRefInvoker`, `FixedOutput`, plus five test fakes — **three
of them real backends.** Its neutrality is **already a measurement, not a
claim.**

**The trap survives, aimed correctly:** `reference_from_registered_recipe` and
`reference_output` take `&CudaDevice` with **no trait at all**. So *"one
implementor on both sides launders a dependency as an abstraction"* applies to
the **reference realizer**, which has zero abstraction — not to `KernelInvoker`,
which has nine siblings. **A general rule was stated and pointed at the one role
where it does not apply.**

**This is GAP-236 and it is CireSnave's decision, not this design's**, but the
restructure changes its shape: publishing a provider-facing surface is a very
different act when it lives in a crate about to be renamed and republished.
**Whatever is decided should be decided before Stage 4, not after.**

And the trap Vulkane caught stands regardless: **a trait with one implementor
used on both sides of a comparison launders a dependency as an abstraction.** If
the verify seam is published, the two device roles need *structural or observable*
independence — different traits, or a verdict that records which implementation
realized the reference.

---

## 7. Gates

Each stage is gated, and the gates are chosen so that a false green is hard:

1. **Facade path-set diff** — the public path set under `fuel::` is byte-identical
   before and after every move. This is the load-bearing gate and it is
   mechanical.

   ⚠️ **AND A PATH-SET DIFF IS NOT SUFFICIENT — Lightbulb's catch, 2026-08-26, on
   a gate that would have encoded a false claim.** A path diff compares **names**.
   A re-export resolving to a **different type** at the same path passes it:
   `fuel::Device` could point at `fuel_core::device::Device` before and a
   newly-written `fuel_hardware::Device` after, and **Fuel compiles while a
   downstream `impl From<CudaDevice> for Device` silently targets a different
   type.** The gate as first written claimed *API stability* and delivered *name
   stability*.

   **Closing it, cheapest first:**

   - **Type-identity assertions for every type an external consumer names** — one
     line each, compile-time, no runtime cost:
     ```rust
     const _: () = { fn _same(x: fuel::Device) -> fuel_hardware::Device { x } };
     ```
     Compiles **only** if the two are literally the same type. Add one per
     externally-named type at Stage 1, before anything moves.
   - **Diff rendered item signatures with fully-qualified defining paths**
     (`cargo public-api`), not the bare set of paths.
   - **A downstream compile fixture is what actually closes it.** The mechanical
     gates narrow the hole; only compiling a consumer's usage against the facade
     shuts it. Saying that plainly is better than claiming the diff covers it.
2. **`fuel-examples` compiles untouched** — 553 `fuel::` imports, the heaviest
   consumer in the tree, and per the working agreement it is a *constructing*
   crate for most model weights structs, so it catches field-level breakage a
   `--lib` gate cannot see.
3. **`--all-targets` per moved crate**, not `--lib`: a move that compiles the
   library and breaks `tests/` is the documented failure mode.
4. **CI stays green across all 10 jobs.** It went green for the first time in 995
   runs tonight; a restructure that reds it is not "temporarily" red.
5. **No `cargo check --workspace` claim without naming the exclusions** —
   `fuel-metal-backend`, `fuel-metal-kernels` (wrong platform),
   `fuel-mkl-cpu-backend`, `fuel-aocl-cpu-backend` (missing SDKs),
   `fuel-cuda-backend` (unconditional kernel forge).

**Not a gate, deliberately**: "the tests pass." Every stage here is a move, and a
move that preserves behaviour while breaking a boundary passes its tests. The
path-set diff is what catches that; the test suite is not.

---

## 8. What this deliberately does not propose

- **No `fuel-autograd` split** — no demonstrated consumer (§5, Stage 3).
- **No 147-crate model explosion** — rejected by the ratified stopping rule.
- **No `fuel-error` up front** — it lands when two crates need it and cannot
  share a nearer home.
- **No changes to `fuel-dispatch` in this restructure — but the reason first given
  here was WRONG, and correcting it is worth more than the conclusion.**

  ⚠️ **This bullet used to read:** *"dissolving it is not on any critical path.
  The stopping rule cuts both ways: **absence of a consumer boundary** is a reason
  not to split."* **Measured 2026-08-27 after CireSnave asked why `fuel-core`
  dissolves and `fuel-dispatch` does not: there IS a consumer boundary, it is a
  quarter of the crate, and it has two named external would-be consumers.**

  ```
  fuel-dispatch/src/   115,052 lines   (138,772 including tests/ and benches/)

    fkc/                30,171   26.2%   35 files   <- kernel contracts + verification
    pipelined.rs        15,758   13.7%              <- the executor
    dispatch.rs         13,287   11.5%
    ranker/              9,703    8.4%   20 files
    vulkan_dispatch.rs   8,426    7.3%
    baracuda_dispatch.rs 6,881    6.0%
    plan.rs              5,135    4.5%
    jit_ingest.rs        4,195    3.6%
    telemetry/           4,158    3.6%   10 files
  ```

  **`fkc` is 26% of the crate, and a kernel PROVIDER wants contracts and
  verification without wanting the executor.** That is a real class — **Vulkane
  and Unpopped are both in it** — and it is precisely what **GAP-236** is about.

  **The correct reasons it does not dissolve *in this restructure*:**

  1. **No publish blocker.** `fuel-dispatch` is **free** on crates.io (measured,
     with a negative control). `fuel-core` is a 125-version blockchain client —
     **that is the forcing function, and it does not apply here.**
  2. **02-layers does not ratify it as dissolving.** It ratifies `fuel-core`
     dissolving and names `fuel-dispatch` as a Foundation crate with a defined
     role. **Dissolving it would be a NEW architectural decision, not the
     execution of an existing one** — and this document's whole premise is that
     it is a path to a ratified destination, not a redesign.
  3. **It is coherent in the way `fuel-core` is not.** `fuel-core` is **73% model
     zoo** — content sitting in a framework crate. `fuel-dispatch` is dispatch
     machinery and kernel contracts: **all of it framework.** Size is not the
     argument; composition is.

  **So the honest statement is: `fuel-dispatch` is not dissolving, but it may
  SPLIT — and the split is gated on GAP-236, which is CireSnave's open
  decision.** Publishing `fkc::verify` as a provider-facing surface is the thing
  that would justify pulling `fkc` out; **until that is decided, extracting it
  would be building a crate boundary for a consumer we have not agreed to have.**

  **Note what the wrong reason would have cost:** *"no consumer boundary"* is an
  absence claim, and it was made without measuring the composition. Had it stood,
  the restructure would have carried a stated finding that the crate is
  indivisible — into exactly the period when GAP-236 gets decided.
- **No trait-only crates.** CireSnave corrected this explicitly: *trait-centered*
  and *trait-only* are not the same thing, and nothing here proposes the latter.

---

## 9. Open decisions

| # | Decision | Owner | Blocking |
|---|---|---|---|
| 1 | ~~`fuel-model` name collision~~ **SETTLED — `fuel-model` IS the authoring surface.** Resolved with CireSnave in the same conversation that raised it: the ratified tier is `fuel-model-core` + `fuel-model-*`, and **a bare `fuel-model` was never defined by 02-layers**, so it was adjacency confusion rather than a collision. **And the adjacency is CORRECT once the relationship is stated: `fuel-model-*` crates are WRITTEN IN the `fuel-model` language, so naming them as its leaves is accurate.** Zero renames, no ratified tier touched, and the language is Rust (build-time), so **no `fuel-model-macros` proc-macro crate is required** — an ergonomics call for later, not a structural one now. ⚠️ **This was settled and then left on the portfolio PM's decision queue for hours because the architect never went back to close it** — see §10. | CireSnave | ✅ closed |
| 2 | Registry availability for every new crate name | — | Stage 4 publish |
| 3 | Judge's destination (§5, Stage 3) | architect | Stage 3 only |
| 4 | Serving/decode destination (§5, Stage 3) | architect | Stage 3 only |
| 5 | GAP-236 / publish `fkc::verify` | CireSnave | Stage 4 |
| 6 | ~~Does Lightbulb import via `fuel::` or `fuel_core::`?~~ **ANSWERED 2026-08-26 — `fuel_core::` = 0, `fuel::` = 71, positive-controlled (the same `git grep -c` returns 71, so the zero is real). Lightbulb is in the 98.3% and has NO PORT under the facade lever.** Their other direct roots, measured rather than assumed: `fuel_inference::` 2 (tests only), `fuel_cuda_backend::` 2 (1 production, `device.rs:25`), and `fuel_graph::`/`fuel_ir::`/`fuel_dispatch::` **comment-only — 5 sites, zero code**, named explicitly because they would inflate a census. | Lightbulb | ✅ closed |

**NOTHING NOW BLOCKS STARTING.** #1 was settled in conversation, #6 closed by Lightbulb's measurement, and #2 (registry availability) is measured — every candidate name is free except `fuel` and `fuel-core` themselves, which costs a published name rather than the design (§4). Stages 1 and 2 — the lever and 73% of the mass —
depend on nothing in this table except the name of a crate that does not yet
exist, and Stage 1 does not even need that.

---

## 10. Provenance

This document's four inputs each came from someone measuring rather than
agreeing, and each made it smaller:

- **Vulkane** measured the verify seam instead of asking about it, and caught that
  the symmetric-trait fix does not achieve independence.
- **Lightbulb's** dependency being a trait impl rather than a call was the PM's
  catch, and it converted a derivable detail into a question the design must
  answer.
- **The `fuel-core` blast radius** turned out to be 6 crates, and **73% of the
  dissolving crate is content, not framework** — both measured here, both
  contrary to how the job had been described.
- **Three CLAUDE.md claims are false at head** (§2), found incidentally while
  establishing the current state.

- **Vulkane** ran Fuel's never-executed `VulkanInvoker` on real hardware
  unprompted (it works — `add_f32` over Vulkan-resident storage is correct), and
  their **positive control** found that the test passes with **no Vulkan device
  at all**: GAP-243, a class of **49** such tests across 13 files that nobody had
  enumerated.
- **Lightbulb** answered §9 #6 from measurement with a positive control, and
  caught that the load-bearing gate compared names rather than types.
- **The portfolio PM** checked registry availability for every name and found
  `fuel` itself occupied — the one name the central lever is built on.

- **A decision of the author's own aged into a false one.** The `fuel-model` name was
  **settled with CireSnave in the conversation that raised it**, and the architect had
  already told the portfolio PM it was open with a different recommendation. **He never
  went back.** It sat on CireSnave's decision queue under a wrong label until *he* asked
  *"I thought that was settled — was it not?"* **A blocked-on record with no expiry ages
  into a confident lie**, and the person best placed to notice is the one who created it
  and therefore never re-reads it.

The single most consequential fact in this document — **1018 `fuel::` versus 18
`fuel_core::`** — was a two-second query that nobody had run, and it turns the
riskiest-looking stage into the safest one.
