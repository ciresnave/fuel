# fuel-core dissolution — B1 scoping (2026-09-24)

> Started as a SCOPING document with no code moved. That premise is now partly stale by
> design: Slice 2 (`fuel-loaders`, #240), the `fuel-model-llama`/`fuel-model-phi` extraction,
> and both crates' shim-debt ledger entries below are executed, not proposed. Slice 3
> (`safetensors.rs`) is scoped and held pending a separate MLMF architectural question. It
> supersedes the "remains downstream of B6" framing at the end of `fuel-core-retirement-b0.md`
> and the 2026-08-19 amendment beneath it — both are now stale in ways this doc corrects with
> evidence, not just a date stamp.

## Why this doc exists

CireSnave pushed back on a board item that framed `fuel-core`'s crates.io name collision
(with FuelLabs' blockchain client) as a rename decision: *"Fuel-core was supposed to be
dissolved and all code within it distributed to other crates. Why would we be back to
looking at publishing fuel-core?"* He was right — if `fuel-core` dissolves, there is
nothing to publish under that name and nothing to rename. That reopened the dissolution,
which `fuel-core-retirement-b0.md`'s own 2026-08-19 amendment had already flagged as
**unscoped, not blocked**: B0 (the fuel-ir/fuel-hardware/fuel-backend-contract/
fuel-cpu-kernels extraction) is complete; B6 (eager-dispatch retirement, the stated
blocker) is complete; nobody wrote the steps for what remains.

This doc is those steps — as far as they can be written without moving code.

> **Update (2026-09-24, after Slice 2 executed in #240):** the line above was true when this
> doc was drafted; Slice 2 has since moved real code (`fuel-model-loader`, PR #240). Left
> unedited above as the historical record of the doc's own starting state — see the
> definition-of-done section immediately below, which #240's shims made necessary.

## Definition of done, and the shim-debt ledger — added 2026-09-24, before Slice 3

**This section did not exist when Slice 2 executed, and its absence was itself a finding.**
Slice 2 (#240) moved 8 files into `fuel-model-loader` and left `fuel-core`'s originals as
re-export shims — the same pattern already used for `dtype.rs`/`layout.rs`/`shape.rs`/
`backend.rs`/`storage.rs` from the earlier B0 extraction. Reporting that precedent as
justification surfaced the actual problem: **those B0-era shims are still in place, months
later. The shim-removal step has never once been taken in this project.** If every slice
of this dissolution leaves a shim behind and no slice ever removes one, the code moves and
`fuel-core` does not dissolve — it becomes a permanent facade of re-exports, five
dependents deep, which is a different and never-decided outcome from *"fuel-core dissolved
and all code within it distributed to other crates."*

**Answer, grounded in CireSnave's own words quoted above, not invented here:** he said
*dissolved*, not *reduced to re-exports*. A crate of shims does not meet that bar. **The
target is deletion of `fuel-core`, not an indefinite facade** — every shim recorded below
is deferred work with an owner and a trigger, not a resting state.

⚠️ **RULED (2026-09-24), verbatim, not the inference above:** *"Fuel-core will be deleted.
The functionality within it needs to be moved out of it into more proper homes."* The
inference was correct and is superseded by this quote, not merely confirmed by it — from
here on cite the ruling, not the reasoning that guessed it.

⚠️ **Consequence: the shim-debt ledger below is no longer a record of what happened. It is
a work plan — every row is scheduled work, not accepted state**, because `fuel-core` cannot
be deleted while a shim lives in it. This includes the B0-era shims, which were flagged but
deliberately left uncounted in the first version of this section; they are now in scope on
the same terms as everything else and are re-measured below rather than carried forward
uncounted.

⚠️ **"More proper homes" is a design constraint, not just a destination.** He did not say
move it anywhere out of fuel-core — a slice with no coherent target crate and a stated
responsibility is not done, it has just relocated the junk-drawer problem. `fuel-model-loader`
(Slice 2, tier 45, "IO / formats", beside `fuel-formats` at 30) is the precedent to match. If
a future slice has no proper home, that is a finding to report, not a reason to invent a
`fuel-misc`.

⚠️ **Board item 52 (the `LlamaModel`/`PhiModel` placement question) is now BLOCKING, not
optional, because of this ruling — filed as such, not resolved here.** Under a facade
reading those could have stayed in `fuel-core` behind a shim indefinitely; under DELETE they
must go somewhere, and nobody has yet said where. Still not blocking Slice 3.

⚠️ **This doc is now an interface contract, not working notes.** A Lightbulb port onto
Fuel's public API is expected soon (CireSnave: *"Candlelight is extremely inferior to
Fuel"*), which means external consumers may start depending on paths this dissolution is
actively moving. Keep this doc current on `main`, not stale in a branch — and **flag any
shim removal to the PM before executing it**, since removing a shim is a breaking change for
anyone who has already adopted the old path.

### What deletion requires, stated once so every slice can point at it

`fuel-core` deletes only when both are true:
1. **Every real implementation has moved** to its target crate (the slice work).
2. **Every external consumer imports the target crate directly** — no `fuel_core::` /
   `fuel::` path resolves through a re-export shim anymore. This is the step no prior
   slice has taken.

(2) is the expensive, mechanical, low-risk-per-edit / large-count part — the ~200-call-site
repoint flagged as "large but mechanical, not a semantic risk" for Slice 2's cluster is a
preview of what this looks like at full scale, times however many slices land shim-first.

### The consumer-repoint programme — scheduled, not implied

**This is a real, separate phase of the dissolution now, not an assumption that lived only
inside the DELETE ruling.** Sizing, measured against the ledger below, not guessed:

- **Slice 2's cluster:** ~55 files (30 for `hf_config`, 0 for `model_progress`, ~24 for
  `quantized::*`).
- **Slice 3's cluster (`safetensors.rs`):** 110+ files.
- **B0-era shims:** re-measured below — smaller in file-count than Slices 2/3 by naive
  qualified-path grep, but that instrument is known to undercount (see the row) because most
  consumers import the symbol once (`use fuel::{DType, Shape};`) and use the bare name
  throughout; a reliable count needs a compiler-based census, not a text search.
- **Running total, lower bound: 190+ files**, before the B0-era true count is known and before
  any future slice (the still-unscoped ~2.3 MB) adds its own consumers.

**Sequencing choice: land slices shim-first (lower risk per slice, matches what #240 and this
Slice 3 do) and do consumer repoints as a separate, later sweep per shim — not one repoint per
slice.** Reasons this is a program-level decision rather than a per-slice one:
1. **Batching amortizes the review cost.** A 200+-file import-path rewrite reviewed one slice
   at a time is 200+ small mechanical diffs across several PRs; reviewed as one sweep per
   shim, a reviewer checks the mechanical pattern once and spot-checks the rest.
2. **Sequencing against Lightbulb matters more than sequencing against slices.** Per the PM
   (2026-09-24): a Lightbulb port onto Fuel's API is expected soon, and shim removal is a
   breaking change for anyone who has adopted the old path. The repoint sweep needs to be
   timed against THAT migration, which slice-by-slice repointing cannot do — a sweep is a
   single event that can be scheduled; N per-slice repoints are N events to coordinate.
3. **It can be done incrementally, shim by shim, once scheduled** — nothing requires doing
   all ~200+ files in one PR. The unit of the sweep is one shim's full consumer set (e.g., all
   30 `hf_config` files in one PR), not the whole ledger at once.

**Trigger, stated so it isn't implicit:** the sweep for a given shim starts when the PM
clears it against the Lightbulb port's current state — not automatically when a slice lands.
Until then, the shim stays, and that is compliant with the DELETE ruling as long as it is
tracked here, not silently accepted.

### Shim-debt ledger

| Shim (in `fuel-core`) | From slice | Real home | Consumers to repoint | Repoint requires |
|---|---|---|---|---|
| `src/hf_config.rs` | Slice 2 (#240) | `fuel_model_loader::hf_config` | 30 files, all `fuel-transformers/src/models/*.rs` + 1 internal (`fuel-core/src/lazy.rs`) | `fuel_core::hf_config::` → `fuel_model_loader::hf_config::` import rewrite; `assert count==1` per anchor per the standing enumeration discipline |
| `src/model_progress.rs` | Slice 2 (#240) | `fuel_model_loader::model_progress` | **0 external** — purely internal to the moved cluster | Delete the shim outright once nothing references it; no repoint needed, easiest item on this list |
| `src/quantized/mod.rs` (+ its re-exported `arch`/`gguf_file`/`gguf_mmap`/`imatrix_file`/`tokenizer` submodules) | Slice 2 (#240) | `fuel_model_loader::quantized::*` | ~24 files: `fuel-transformers` quantized model variants, `fuel-tensor-tools`, `fuel-examples` quantized examples, `fuel-lazy-examples` | Same import-rewrite pattern; note 3 grep hits on `crate::quantized` in `fuel-backend-contract`/`fuel-ir` are FALSE POSITIVES (their own unrelated `quantized` modules) — re-verify this at repoint time, don't carry the exclusion forward from memory |
| `src/dtype.rs`, `src/shape.rs` (root re-exports `DType`, `Shape`, `D`) | B0 (2026-06 era) | `fuel_ir::dtype`, `fuel_ir::shape` | **Re-measured 2026-09-24, LOWER BOUND, not exact**: qualified-path grep (`fuel_core::DType`/`fuel::DType`) finds 6+8=14 files; import-line grep (`use fuel(_core)?::{...DType...}`) finds 20. `Shape`: 1+13 qualified, 50 via import line. ⚠️ **Both instruments undercount**: a grouped multi-line import (`use fuel::{\n DType,\n Shape,\n};`) defeats a single-line regex, and once imported, most call sites use the bare name — which this row cannot distinguish from an unrelated same-named symbol in another crate. **Reliable count needs a compiler-based census** (CLAUDE.md's own standing rule: "use the compiler as the caller census, not a grep") — e.g. temporarily delete the shim and read the resulting `E0432`/unresolved-import list, which cannot false-positive on a same-named symbol elsewhere. Not done in this pass. |
| `src/layout.rs`, `src/storage.rs`, `src/strided_index.rs` (root re-exports `Layout`, `Storage`, `StridedIndex`, `StridedBlocks`) | B0 | `fuel_ir::layout`, `fuel_backend_contract::Storage`, `fuel_ir::strided_index` | **0 by both instruments above** — genuinely low or zero real usage, OR the same undercount applies and these are simply rarer symbol names with less collision risk (both readings are consistent with a 0; the compiler census would disambiguate). Not claimed as verified-zero. |
| `src/dyn_backend.rs`, `src/backend.rs` (no root re-export; module-path only) | B0 | `fuel_backend_contract::{dyn_backend, backend}` | `dyn_backend`: 1 file (module-qualified grep, no root-export ambiguity so this one IS reliable). `backend`: 0 files. | These two are the one part of the B0-era row that qualified-path grep CAN answer reliably, because nothing re-exports them at the crate root to create a bare-name ambiguity. |
| `src/safetensors.rs` | Slice 3 (open PR) | `fuel_model_loader::safetensors` | 110+ files across `fuel-transformers/src/models/` and `fuel-nn/src/` (measured in #239/#240's review; not yet re-verified per-module the way Slice 2's cluster was) | Largest single repoint on this ledger by consumer count; `MmapedSafetensors`/`BufferedSafetensors` have their own real implementation (not a shim over `fuel-formats`, unlike its sibling `gguf_file.rs`/`imatrix_file.rs`) |

**Trigger for acting on this ledger:** none, yet — that omission is deliberate, not an
oversight, and stated so the omission itself doesn't quietly become the next stale record.
The repoint sweep is real, mechanical work that should be scheduled once the shape of the
FINAL crate boundary is known (i.e., once the Llama/Phi placement question on CireSnave's
board is answered — that answer determines whether `fuel-core` ends up empty enough that a
full sweep is worth doing in one pass, or whether it happens per-shim as each one's slice
lands). **Do not let "no trigger yet" become "no trigger ever" — this row set is the thing
to re-read before this doc is next touched.**

## Correction 1: the amendment's "first slice" has already shipped

The 2026-08-19 amendment named `fuel-nn` (22 files / ~8,855 lines, zero eager-`Tensor`
dependency) as the first separable piece and recommended extracting it. **That extraction
already happened, undocumented, on 2026-09-02.**

Re-measured at `origin/main` (`a732c431`): `fuel-nn` exists as a top-level crate — 22
source files, 9,002 lines. Its own `Cargo.toml` says so directly:

> *"Sits ABOVE fuel-core, not beside it... The fuel-core dissolution repoint has run
> (2026-09-02): fuel-nn imports `fuel_core::` directly rather than through the `fuel`
> facade — one crate's `use` lines, exactly as anticipated."*

**Do not re-propose this slice.** The reason this doc exists — a scoped-but-unexecuted plan
going stale in the other direction — has already happened once. Update
`fuel-core-retirement-b0.md`'s amendment to say DONE rather than leave it reading as
pending; a stale "TODO" next to a shipped item is exactly the failure mode CireSnave's push
was named after.

## Correction 2: `fuel-formats` already exists and fuel-core partially delegates to it

Before proposing any new crate, checked whether one already covers this ground.
`fuel-formats` exists, with an explicit design contract: *"Transport-independent parsers
for tensor wire formats... No item in this crate references `Tensor`, `Device`, `Storage`,
or any other backend-frontend type."* It covers `safetensors`, `pickle`, `gguf`, `ggml`,
`imatrix`.

`fuel-core` already depends on it and two of its own modules are genuinely thin delegation
shims, not original implementations:

- `fuel-core/src/quantized/gguf_file.rs` — re-exports + delegates to `fuel_formats::gguf`.
- `fuel-core/src/quantized/imatrix_file.rs` — delegates to `fuel_formats::imatrix::load_path`.

**`fuel-core/src/safetensors.rs` is NOT a thin shim, and an earlier draft of this doc
overstated it as one — corrected on review, not by me catching it.** Only its `MmapedFile`
type re-exports `fuel_formats::safetensors::MmapedFile`. The rest of the file —
`MmapedSafetensors` and `BufferedSafetensors`, real wrapper types with their own
implementations — is original, and has a broad public consumer surface: `git grep -c` for
those two type names returns nonzero hits in over 110 files across `fuel-transformers/src/models/`
and `fuel-nn/src/`. What IS still true and checked independently: neither wrapper touches
`crate::lazy::Tensor` (confirmed by grep, zero references to `crate::lazy` in the file) —
their `get`/`tensors` methods return the `safetensors` crate's own `TensorView`, which is why
this file still belongs in the zero-`Tensor`-coupling slice below. It is just not a small or
"already done" part of that slice — moving it means moving real, heavily-depended-upon
implementation, not deleting a shim.

**So the byte counts below are gross sizes, and only PARTIALLY overstate the remaining
work** — `gguf_file.rs`/`imatrix_file.rs` are already-done shims (make the task smaller
than their bytes suggest), but `safetensors.rs`'s bytes are close to the real migration
cost (large public API surface to re-point, not a shim to delete).

### The pattern across both corrections, worth naming once rather than twice

Correction 1 (`fuel-nn`) and Correction 2 (`fuel-formats`) are the same defect from two
directions, and a third instance of it landed on CireSnave's own board within an hour of
this doc's first draft (a board item citing `fuel-nn` as the pending first slice, written
after the extraction had already shipped — corrected the same day it was written). **A
stale record does not just fail to tell you work is now possible; it can send you to
redo work already done, or to size a task by a byte count that measures the wrong
object** (a file's total size, when part of that file is already a shim over a crate
built after the count was last taken). Neither failure announces itself — both read as
authoritative right up until someone re-measures. The fix in both directions is the same
one this doc tries to model throughout: **re-derive the fact from the current tree before
citing it, and print the ref it came from**, rather than trusting a doc's own account of
its subject's state — including this doc, the day someone next reads it.

## What is genuinely re-measured and still unscoped: `fuel-core`'s current shape

Re-measured directly (not carried from any prior doc), at `origin/main` (`a732c431`):

    fuel-core/                84 files, 2,430,392 bytes
    fuel-core/src/lazy.rs     1 file,   1,109,731 bytes  (46% of the crate, alone)

Four real (non-dev) dependents: `fuel` (facade), `fuel-nn`, `fuel-datasets`,
`fuel-transformers` — all `[dependencies]`. `fuel-vulkan-backend` depends on it too, but
only under `[dev-dependencies]` (test-only, not a production coupling).

## `lazy.rs`: NOT a mechanically-splittable monolith — say so plainly, as asked

Structural read of the file (top-level item positions, not a guess): the first ~8,000
lines (of ~24,000 total, by line-position of top-level items) are the `Tensor` type and
its inherent impls. **Everything after that — roughly the back half of the file — is the
`LlamaModel`/`LlamaWeights`/`LlamaConfig`/`LlamaTokenizer` and
`PhiModel`/`PhiWeights`/`PhiConfig` implementations**, defined in the same file as `Tensor`
itself.

This is not neglect. `LlamaModel` is *the* canonical base every downstream Llama-family
model builds on, confirmed from the callers' own doc comments, not assumed:

- `fuel-transformers/src/models/lazy_llama_full.rs`: *"`lazy::LlamaModel` is the canonical
  lazy-graph LLaMA decoder used..."* and wraps it directly.
- `fuel-transformers/src/models/lazy_llama2c.rs`: *"Thin wrapper over
  `fuel_core::lazy::LlamaModel`."*
- `lazy_llava.rs`, `lazy_mistral.rs`, `lazy_deepseek2.rs` all reference or mirror
  `fuel_core::lazy::LlamaModel`'s methods directly in their own doc comments.
- `fuel-core/src/inference_context.rs` (persistent/paged decode machinery) links directly
  to `crate::lazy::LlamaModel::forward_paged_step_persistent` and
  `forward_decode_step`.
- `fuel-inference/src/multi_session.rs` calls
  `fuel::lazy::LlamaModel::generate_streaming_with_kv_context` directly, and states its own
  scope as adding "no [new logic]" beyond what `inference_context` + this `LlamaModel`
  already provide.

**Moving `LlamaModel`/`PhiModel` out of `lazy.rs` is therefore not a mechanical slice.** It
would require deciding where a *canonical base model that `fuel-transformers` wraps* is
supposed to live relative to `fuel-transformers`' own model zoo (which already holds
`lazy_llama_full.rs`, `lazy_phi.rs`, `lazy_phi3.rs`, and their quantized variants) — a real
architectural question, not a scoping one. **Flagging this here rather than producing a
plan that quietly assumes it splits cleanly**, per the stop condition on this task. This
needs the architect's ruling before anyone sizes it as a slice.

## `fuel-model-llama` / `fuel-model-phi`: executed, ratified by `02-layers.md`

The stop condition above is resolved: `docs/architecture/02-layers.md` already ratifies
`fuel-model-llama`† and `fuel-model-phi`† (dagger-marked, one architecture per crate), and
CireSnave authorized starting the extraction. Executed as PR (this change).

**An ordering law this extraction discovered, generalizing past this one slice:** a
deferred re-export shim (the pattern every other slice in this doc uses) is only available
when extracting DOWNWARD in the tier file. Slices 2/3 worked because `fuel-loaders` depends
only on crates strictly BELOW `fuel-core` and never back up to it, so `fuel-core` could
`pub use fuel_loaders::*;` without a cycle. `fuel-model-llama`/`fuel-model-phi` sit ABOVE
`fuel-core` (tier 110 > 100) and their own consumers reach them THROUGH `fuel-core`'s path
today — so `fuel-core` re-exporting them back would be `fuel-core → fuel-model-llama →
fuel-core`, a hard cargo cycle, not a style choice. **Any future slice whose consumers
currently reach it via `fuel_core::X` must check this before assuming a shim is available.**

**Consequence: no shim. Every real consumer was repointed in this same PR**, not deferred.
Measured via a multiline-aware scan (a single-line grep undercounted twice in a row before
converging — 9 → 22 → 24 files, the middle count wrong because `rustfmt` wraps long import
lists across lines, invisible to a same-line pattern): **24 files** — 15 `fuel-examples`/
`fuel-lazy-examples` binaries (low-risk), 4 `fuel-transformers` model files, 1
`fuel-inference/src/multi_session.rs` (real: `impl DecodeModel for LlamaModel` /
`impl PagedDecodeModel for LlamaModel`, orphan-rule-legal since the traits are local to
`fuel-inference`), and 2 test-only sites in `fuel-core` itself (see below). 4 more files
carried doc-comment-only mentions (stale intra-doc links, fixed for hygiene, no compile
impact). One false positive worth recording: `fuel::lazy_phi::PhiModel` in
`fuel-examples/examples/phi/main.rs` is `fuel-transformers`' OWN, separately-defined
`PhiModel` (declared in `lazy_phi.rs`) — a name collision with `fuel-core`'s `PhiModel`, not
a reference to it. Same shape as the `quantized` module collisions Slice 2 caught.

**A second finding, inside `lazy.rs` itself: textual adjacency to `LlamaModel`/`PhiModel`
was not evidence of ownership.** Several types/functions positioned in the file's "LLaMA
section" — `WeightStorage`, `LayerWeights`, `SamplingStrategy`, `LayerNormPair`,
`ConvWeightBias`, `load_tensor_as_f32`, `load_transposed_matrix(_preserve_dtype)`,
`apply_affine_rms_norm`, `sample_logits` (+ its private `spec_*`/`sample_multinomial`
helpers), `build_decode_causal_mask(_windowed)`, `offer_flash_decode_arm_for_region`,
`invalidate_decode_pair_if_stale`, `refresh_decode_session` — are GENERIC, with consumer
counts from a handful up to **~120 files** across the whole `fuel-transformers` model zoo,
and two (`build_decode_causal_mask*`, `refresh_decode_session`) are called by `fuel-core`'s
own `persistent_decode.rs`. **Moving the "LLaMA section" wholesale would have made
`fuel-model-llama` a de facto shared-utilities dependency for the entire model zoo — the
exact inverse of "one architecture per crate."** All of it stayed in `fuel-core`; the two
new crates call back into `fuel_core::lazy::X` for it (several items' visibility widened
from `pub(crate)`/private to `pub` to make that legal — no behavior change, `cargo check`
catches every miss). `TokenDataHost`/`TokenDataBytes`/`captured_output_to_f32` are the
narrower case: shared between `LlamaModel` and `PhiModel` specifically (not the wider zoo),
same treatment.

**A third finding: two fuel-core tests couldn't dev-depend their way around the cycle
either.** `fuel-core/src/decode_shape.rs`'s own `#[cfg(test)]` module and
`fuel-core/tests/paged_decode_parity.rs` construct a real `LlamaModel` to test
`fuel-core`-internal predicates against it. A `[dev-dependencies]` entry on
`fuel-model-llama` compiles (Cargo permits dev-dependency cycles — they sit outside the
normal library build graph) but hits a DIFFERENT wall at the type level: the test binary
ends up with two non-unifying copies of `fuel_core` in its dependency graph (the one being
tested, and the one `fuel-model-llama` depends on), so a `fuel_core::decode_shape::
ModelInstanceId` constructed in the test and one expected by `fuel_model_llama::
LlamaWeights` are reported as different types. **Fix: moved both tests out of `fuel-core`
into `fuel-model-llama`'s own `tests/`** (`decode_shape_geometry.rs`,
`paged_decode_parity.rs` — `git mv`, zero content change needed for the latter since it had
no `crate::` references). There `fuel_core` is an ordinary single dependency; no diamond.
Correctly re-scoped as integration tests of "does `LlamaModel` integrate with `fuel-core`'s
decode machinery", which is what they actually test now that `LlamaModel` lives elsewhere.
**No `[dev-dependencies]` entry on `fuel-model-llama` was needed in the end.**

**Measured, not estimated, per the gate's own request — three numbers, this PR's diff:**

    fuel-model-llama/src/lib.rs   12,317 lines  (structs/impls/tests moved from lazy.rs)
    fuel-model-phi/src/lib.rs      3,107 lines  (structs/impls/tests moved from lazy.rs)
    fuel-core/src/lazy.rs         11,231 lines  (Tensor bridge + shared utilities, retained)
    ------------------------------------------------------------------------------
    original fuel-core/src/lazy.rs: 26,580 lines

Roughly 58% of the file's lines moved — smaller than the naive "everything after the Tensor
bridge" estimate this doc's own earlier analysis implied, because the shared-utility carve-out
turned out to be substantial, not a rounding error.

**Shim-debt ledger entries, recorded at creation as this section requires:**

| Crate | Depends on | Why (stepping-stone, not end state) | Removal condition |
|---|---|---|---|
| `fuel-model-llama` | `fuel-core` | Needs `Tensor` bridge + `inference_context`/`kv_block_pool`/`kv_block_pool_device`/`persistent_decode`/`pipelined_bridge`/`decode_shape`/`safetensors`, none rehomed | `fuel-core`'s retained machinery finds a real home |
| `fuel-model-phi` | `fuel-core` | Same as above | Same as above |

**Open finding, NOT acted on here, flagged for CireSnave per the architect's instruction:**
the ~9-function shared-utility layer inside `lazy.rs` (generic tensor-loading/sampling/
decode-mask helpers, used by ~120 files, currently living in `fuel-core` by accident of
history) has no ratified home. `02-layers.md` says generic building blocks (RoPE, RMSNorm,
GQA attention, SwiGLU MLP) belong in `fuel-nn` — these look like the same class, and are
plausibly the single largest remaining piece of this dissolution. Not scoped or sized here;
recorded so it is not silently re-discovered later.

## The one genuinely-scoped, zero-Tensor-coupling next slice

Nine files, re-measured at `a732c431`, checked one at a time for internal imports (not
assumed):

    fuel-core/src/hf_config.rs                 13,529 B   (imports: crate::{Error, Result} only)
    fuel-core/src/model_progress.rs             4,801 B   (imports: none)
    fuel-core/src/quantized/arch.rs             7,040 B   (imports: none)
    fuel-core/src/quantized/gguf_file.rs        1,711 B   (shim over fuel_formats::gguf)
    fuel-core/src/quantized/gguf_mmap.rs        4,782 B   (imports: crate::Result, crate::model_progress)
    fuel-core/src/quantized/imatrix_file.rs       711 B   (shim over fuel_formats::imatrix)
    fuel-core/src/quantized/mod.rs              2,059 B   (imports: none)
    fuel-core/src/quantized/tokenizer.rs       12,058 B   (imports: crate::quantized::gguf_file, crate::{Context, Error, Result})
    fuel-core/src/safetensors.rs                 9,258 B   (imports: crate::{Error, Result}; real MmapedSafetensors/BufferedSafetensors impl, NOT a shim, 110+ consumer files, returns TensorView not Tensor)
    ----------------------------------------------------------------
    TOTAL                                      55,949 B   (9 files)

Every import outside this cluster resolves to `crate::{Error, Result, Context}` —
which are themselves a thin re-export of `fuel_ir::error::{Error, Result, Context}` (checked:
`fuel-core/src/error.rs` is 26 lines, zero internal imports, pure re-export). **A new crate
built from this cluster can depend on `fuel-ir` + `fuel-formats` directly and needs nothing
else from `fuel-core`** — confirmed by grep, not inferred from naming: zero references to
`crate::lazy`, `crate::Device`, or `crate::Tensor` anywhere in these nine files.

**Consumer fan-out is real and large, but mechanical, not risky.** A broad grep (all nine
modules' names in one pattern, not yet disambiguated per-module — flagged as the first
verification step before executing) turns up on the order of 200 call sites: nearly every
`fuel-examples/examples/*/main.rs` and every `fuel-transformers/src/models/lazy_*.rs` uses
HF config parsing or GGUF/safetensors loading. That is expected — config/format loading is
something almost every model example does — and it means the slice is a wide,
find-and-replace-style import repoint (`fuel_core::hf_config::` → `fuel_<new-crate>::` or
similar), not a semantic risk the way the `lazy.rs` split would be. **Re-run the grep
per-module before executing**, since an OR'd pattern across nine names does not tell you
which pattern matched which hit.

### Proposed shape

A new crate — name TBD by the architect, `fuel-model-config` or similar (not `fuel-core`-
anything, to avoid re-creating the naming problem this program exists to solve) — holding
these nine files, sitting **below** `fuel-core` (depends on `fuel-ir` + `fuel-formats`;
`fuel-core` depends on it, not the reverse). `fuel-core`'s own copies become re-export shims
during a transition window (matching the existing pattern from `dtype.rs`/`layout.rs`/
`shape.rs`/`backend.rs`, all already thin shims from earlier B0 work), then get deleted once
every external caller is repointed.

### Verification per this slice (when it executes — not now)

1. Re-run the per-module consumer grep (nine separate greps, not one OR'd pattern) and
   record the disambiguated counts before touching anything.
2. `cargo check -p <new-crate>` standalone — proves it builds with only `fuel-ir` +
   `fuel-formats` as deps, no accidental `fuel-core` edge.
3. `cargo check -p fuel-core --all-targets` with the shim re-exports in place — proves
   nothing broke for internal callers.
4. `cargo check -p fuel-examples --all-targets` and `-p fuel-transformers --all-targets` —
   the two largest external consumer surfaces, both already default-members so this is
   normal-cost, not a special gate.
5. Existing tests at the old and new locations both green (this is a move, not a rewrite —
   no test should need new assertions, only new import paths).

## What is NOT scoped by this pass — say so rather than guess

The remaining ~2.3 MB of `fuel-core` was not investigated to slice-readiness in this pass:
`judge/` (204 KB), `pipelined_bridge.rs` (127 KB), `inference_context.rs` (117 KB),
`kv_block_pool.rs` + `kv_block_pool_device.rs` (175 KB), `train.rs` (71 KB),
`persistent_decode.rs` (64 KB), `scheduling.rs` (58 KB), `nf4.rs` (28 KB — confirmed coupled
to `crate::lazy::Tensor`, not independent), `decode_shape.rs`, `decode_state_spec.rs`,
`telemetry.rs` (self-contained but conceptually tied to the realize path, not the
config/format cluster — its own doc comment ties it to
`pipelined_bridge::build_optimized_graph`), `device.rs`, `factories.rs`, `accelerate.rs`,
and the per-backend `{cpu,cuda,vulkan,metal}_backend/` glue modules. These are the execution
core — almost certainly as tightly wound around `lazy::Tensor` as the model code is, on the
same reasoning that put `LlamaModel`/`PhiModel` next to `Tensor` in the first place. **Each
needs its own evidence-based coupling check before anyone sizes it as a slice**; guessing
from file names here would repeat exactly the mistake this doc is correcting for `fuel-nn`.

## Explicitly out of scope for this doc (per the task that produced it)

- The `Storage`-unification (`fuel_backend_contract::Storage` vs `fuel_memory::Storage`) —
  `fuel-core-retirement-b0.md` already carves this out as its own item.
- Any version bump — the PM allocates at gate time.
- Publishing anything. **`fuel` 0.11.0 must not be published, under any circumstances,
  regardless of anything in this document.**

## Summary for the gate

- Slice 1 (fuel-nn): **already done**, marked so in the B0 doc.
- Slice 2: **executed in #240** — `hf_config.rs`, `model_progress.rs`, and
  `quantized/{arch,gguf_file,gguf_mmap,imatrix_file,mod,tokenizer}.rs` (8 of the 9 originally
  scoped files) moved to the new `fuel-model-loader` crate. `safetensors.rs` deliberately
  held out — not a shim (110+ real consumers), sized as its own follow-up.
  `fuel-core`'s copies of the 8 moved files are now re-export shims. **See the shim-debt
  ledger above — this slice's shims are recorded there with owners, not left implicit.**
- ⚠️ **RULED (2026-09-24), verbatim: *"Fuel-core will be deleted. The functionality within it
  needs to be moved out of it into more proper homes."*** The earlier inference (marked as
  such) is superseded, not merely confirmed. `fuel-core` deletes; it does not become a
  permanent re-export facade. Every shim is now scheduled work, not a resting state.
- **The consumer-repoint programme is its own scheduled phase** (see the new section above):
  lower-bound size 190+ files across Slice 2 + Slice 3 + the reliably-measured part of the
  B0-era shims, done incrementally shim-by-shim, triggered per-shim by the PM against the
  Lightbulb port's timing — not automatically when a slice lands, and not all at once.
- Slice 3 (`safetensors.rs`): scoped, PR open. 110+ consumers, real implementation (not a
  shim like its Slice-2 siblings) — flagged in the ledger as the largest single repoint item.
- ⚠️ **Board item 52 (`lazy.rs`'s Llama/Phi split) is now BLOCKING for the dissolution's
  completion, not optional** — under DELETE, that ~16,000 lines must go somewhere, and
  nobody has said where. Still not blocking Slice 3 or the repoint programme.
- Everything else in `fuel-core` (~2.3 MB): unscoped. **Do not scope it until CireSnave
  answers the Llama/Phi placement question** — that answer changes what those remaining
  files (`inference_context.rs`, `pipelined_bridge.rs`, `judge/`, etc.) are coupled to, so
  scoping them first would risk sizing slices against a boundary that is about to move.
- ⚠️ **This doc is an interface contract for other lanes now, not just working notes** — a
  Lightbulb port onto Fuel's API is expected soon. Keep it current on `main`; flag any shim
  removal to the PM before executing it.
