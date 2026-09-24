# fuel-core dissolution — B1 scoping (2026-09-24)

> This is a SCOPING document, not an execution log. No code has moved. It supersedes the
> "remains downstream of B6" framing at the end of `fuel-core-retirement-b0.md` and the
> 2026-08-19 amendment beneath it — both are now stale in ways this doc corrects with
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

`fuel-core` already depends on it and three of its own modules are already thin delegation
shims, not original implementations:

- `fuel-core/src/quantized/gguf_file.rs` — re-exports + delegates to `fuel_formats::gguf`.
- `fuel-core/src/quantized/imatrix_file.rs` — delegates to `fuel_formats::imatrix::load_path`.
- `fuel-core/src/safetensors.rs` — its `MmapedFile` type re-exports `fuel_formats::safetensors::MmapedFile`;
  its own ~230 remaining lines are raw-bytes mmap/file access returning the `safetensors`
  crate's own `TensorView`, not `fuel_core::lazy::Tensor` — confirmed by grep, zero references
  to `crate::lazy`.

**So the byte counts below are gross sizes, not "originally-fuel-core content" sizes** — a
meaningful fraction of this cluster is already a re-export shim over work already done
elsewhere. The task on this cluster is smaller than its file sizes suggest.

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

Five real (non-dev) dependents: `fuel` (facade), `fuel-nn`, `fuel-datasets`,
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
    fuel-core/src/safetensors.rs                 9,258 B   (imports: crate::{Error, Result}; shim + raw TensorView access)
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

- Slice 1 (fuel-nn): **already done**, mark it so in the B0 doc.
- Slice 2 (this doc's proposal): the hf_config/model_progress/quantized-file-format/
  safetensors cluster, 9 files, ~56 KB, zero measured coupling to `Tensor`/`Device`/`lazy`,
  large but mechanical consumer fan-out. **Authorized (2026-09-24) — proceeds as its own PR
  after this doc, independent of the Llama/Phi question below.**
- The `lazy.rs` Llama/Phi split: **not a slice** until the architect rules on where a
  canonical wrapped-by-`fuel-transformers` base model belongs. Filed on CireSnave's board
  (2026-09-24) as the one part of this dissolution needing a human; not urgent, does not
  block Slice 2.
- Everything else in `fuel-core` (~2.3 MB): unscoped. **Do not scope it until CireSnave
  answers the Llama/Phi placement question** — that answer changes what those remaining
  files (`inference_context.rs`, `pipelined_bridge.rs`, `judge/`, etc.) are coupled to, so
  scoping them first would risk sizing slices against a boundary that is about to move.
