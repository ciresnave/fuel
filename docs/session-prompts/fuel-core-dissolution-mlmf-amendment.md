# Amendment: MLMF owns model-file loading/saving, not another fuel crate

**Status: AMENDMENT ONLY, not executed.** Amends
[`fuel-core-dissolution-complete-plan.md`](fuel-core-dissolution-complete-plan.md) (`docs/session-prompts/fuel-core-dissolution-complete-plan.md`,
135 lines at `5bd79337`, PR #260). That plan is left as-written — this
document does not silently rewrite it — because several of its rows are
about modules (`Tensor`, `Device`, `judge`, `train`, GPU backend bridges)
that have nothing to do with model-file I/O and are unaffected by the rule
below.

## The rule being applied

CireSnave, relayed by the portfolio PM, verbatim: *"Everything directly
relating to Machine Learning Model File loading and saving should be in
MLMF. That means Fuel should be using MLMF for loading and saving any
model-related files."*

This supersedes the original plan's assumption (Part 3, rows for
`hf_config.rs`, `quantized/mod.rs`, `safetensors.rs`) that fuel-internal
loader modules land in another **fuel** crate (`fuel-loaders`). They
should land in **MLMF** instead, with fuel depending on MLMF's published
API rather than owning a second copy of this logic.

## Method

Every module named below was read directly at `origin/main` (`5bd79337`),
not inferred from a file name or from the original plan's descriptions —
several of those descriptions are themselves stale (noted inline where
found; staleness elsewhere is not chased, per [`docs-are-not-code-and-a-sweep-cannot-tell`](../method-rules.md#docs-are-not-code-and-a-sweep-cannot-tell) —
out of scope for this amendment). `fuel-formats/` and `fuel-loaders/` were
enumerated with `git ls-tree -r --name-only origin/main`, not from
`lib.rs`'s `pub mod` list, so nothing private is missed.

**Not duplicating mlmf's own inventory work — and it landed mid-write.**
When this amendment was started, the mlmf lane's
`docs/fuel-migration-inventory.md` was not yet on any ref in
`ciresnave/mlmf` (`gh pr list` empty, `gh api .../branches` checked
directly — no branch held it). It has since landed as `ciresnave/mlmf#101`
(`fuel-migration-inventory` branch, open, not yet merged, 384 lines) —
re-checked before finalizing this amendment, per instruction. **Part E
below reconciles this document against it rather than leaving the "not
pushed yet" framing stale.** This amendment still names *what* moves and
*why* from fuel's side of the boundary; #101 is the authoritative source
for *which mlmf module/crate* each capability lands in and *in what
order* — this document does not restate or re-derive that, it points at
it.

---

## Part A — What changes destination, and why

| Module (fuel, at `5bd79337`) | Original plan said | This amendment says | Why |
|---|---|---|---|
| `fuel-core/src/hf_config.rs` (7-line shim) → real content at `fuel-loaders/src/hf_config.rs`, 13,538 B | `fuel-loaders` (Part 3 row 1) | **MLMF** | It is HuggingFace `config.json` parsing: cross-field-default resolution rules (`head_dim`, `num_key_value_heads`) shared by fuel's per-model config parsers. This is model-file loading logic, not a fuel-specific concern — nothing in it is Fuel's `DType`/`Shape`/`Tensor`, it operates on parsed JSON. |
| `fuel-core/src/quantized/mod.rs` (9-line shim) → real content at `fuel-loaders/src/quantized/{arch,config_from_gguf,gguf_file,gguf_mmap,tokenizer}.rs` | `fuel-loaders::quantized` (Part 3 row 3) | **MLMF**, split by sub-concern — see Part B | The plan's own note that `nf4.rs` couples to `Tensor` by method call, not module path, applies here too in reverse: most of `quantized/` is pure metadata/format parsing (arch detection from GGUF KV metadata, GGUF header/tensor-info reading, tokenizer vocab parsing) with no `Tensor`/`Device` reference — genuinely movable. `config_from_gguf.rs` needs its own read before scheduling (not done in this amendment) since "config from GGUF metadata" sits right at the fuel/MLMF boundary. |
| `fuel-core/src/safetensors.rs` (8-line shim) → real content at `fuel-loaders/src/safetensors.rs`, 9,667 B | **already executed** to `fuel-loaders::safetensors` (Part 3 row 4, PR #259 in flight for consumer repoint) | **MLMF** — a second hop, `fuel-loaders` → MLMF | The original plan treated this as *done*. It moved fuel-core's shim to fuel-loaders' real module, which was correct under the old destination rule but is now itself a shim-in-waiting under the new one. **Do not treat "already executed" as "already correct" — it was correct against a rule that has since changed.** The consumer-repoint sweep in flight (PR #259, batch 1 of 219 landed) should not be read as blocking this: repointing `fuel_core::safetensors::X` → `fuel_loaders::safetensors::X` is still the right first hop (it is real, in-flight, mechanical work against 219 call sites), and a second repoint `fuel_loaders::safetensors::X` → `mlmf::...::X` follows once MLMF's surface exists. Two hops, not a wasted one. |
| `fuel-formats/src/gguf.rs`, 723 lines | not in original plan (that plan only enumerated `fuel-core/src/`) | **MLMF** | Wire-format GGUF parser (magic, KV metadata, `Value`, `TensorInfo`, byte-level reader). Zero `Tensor`/`Device` reference — the crate's own contract doc (`fuel-formats/src/lib.rs`) states this explicitly: "No item in this crate references `Tensor`, `Device`, `Storage`". This is exactly the kind of module CireSnave's rule targets. **Its own doc comment is stale** ("write() of QTensor payloads lives in `fuel-core/src/quantized/gguf_file.rs`") — that file is now `fuel-loaders/src/quantized/gguf_file.rs`; flagged, not fixed here (docs-only amendment, out of scope). |
| `fuel-formats/src/pickle.rs`, 754 lines | not in original plan | **MLMF** | Python-pickle subset parser for `.pth`/PyTorch checkpoints. Same "no backend types" contract. **Its doc comment names a Tensor-construction layer at `fuel-core/src/pickle.rs` that does not exist anywhere in the tree** (`git grep -l PthTensors` finds only this one file) — either dead-code-reaped already or never built; either way there is no live consumer of `PthTensors::get`/`read_all` to repoint, which simplifies this move (no consumer-count sweep needed, unlike `safetensors.rs`'s 219). Confirm zero consumers before moving, don't just trust this reading. |
| `fuel-formats/src/safetensors.rs`, 100 lines | not in original plan | **MLMF** | Thin re-export of the upstream `safetensors` crate's parser surface plus `MmapedFile`. Same contract, same reasoning as `gguf.rs`. |
| `fuel-formats/src/imatrix.rs`, 136 lines | not in original plan | **STAYS IN FUEL — explicit carve-out, do not move** | This is the item HANDOFF flags as already-ruled: CireSnave's item 63a split imatrix between MLMF and Fuel. MLMF's PR #94 (`mlmf-gguf/src/imatrix.rs`, merged) covers the GGUF-*embedded* imatrix wire format only. This file is the OLD raw-binary format (no magic, no version field) — confirmed still in real-world use (bartowski's `Llama-3.2-1B-Instruct.imatrix`, byte-inspected, matches this file's documented layout). Moving it would silently drop support for every raw-format imatrix file in the wild. **Do not fold this into the sweep below just because it lives in the same crate as the other three `fuel-formats` files being moved — it is the one deliberate exception, per an already-made ruling, not an oversight.** |
| `fuel-loaders/src/model_progress.rs`, 4,801 B | `fuel-loaders`, Part 3 row 2, "0 consumers, delete outright" | **Flag, do not decide here** | This is a progress-callback mechanism for loaders (`Fn(&ProgressEvent)`), not a format parser. Its own doc comment says it was deliberately *not* delegated to MLMF already: "Inspired by MLMF's `ProgressFn`/`ProgressEvent` design (`mlmf/src/progress.rs`), cherry-picked to avoid pulling in candlelight as a transitive dependency." That's a real prior objection, not an oversight — but it was written against MLMF's old monolithic `src/` tree (the pre-rewrite root crate, distinct from the newer modular `crates/mlmf-*` layout, which reads far leaner). Whether the weight objection still holds against the new `crates/mlmf-*` structure is unmeasured here — a question for the mlmf inventory coordination, not a call this amendment makes unilaterally. |
| Name-mapping logic | HANDOFF names this explicitly as in-scope | **No generic fuel-side module exists to move** | Searched (`git grep -rniE "name.mapping|name_map"` across `fuel-loaders/`, `fuel-formats/`, `fuel-transformers/`) — zero hits. What fuel has instead is per-model `#[serde(rename = "...")]` attributes scattered across ~10 files in `fuel-transformers/src/models/*.rs`, each mapping one architecture's HF JSON config-field names onto that model's own Rust config struct. This is not the same thing as MLMF's `name_mapping.rs`/`smart_mapping.rs` (tensor-name remapping across GGUF/safetensors/HF checkpoint conventions) — it is config-*field* renaming, tied one-to-one to each model's hand-written Rust struct, and there is no single module to relocate. **Not proposing a move here**: judgment call, not a blind sweep — flagged for the mlmf lane to confirm there is genuinely nothing to consume on fuel's side, rather than asserting it from this grep alone. |
| Saving | HANDOFF asks whether fuel has any save path at all | **No — confirmed, not "not covered by #260"** | B6 (the eager-era retirement) removed the free `load`/`load_buffer`/`save` functions and the `Load`/`View` trait impls from what is now `fuel-loaders/src/safetensors.rs` — its own doc comment states this: "The EAGER half was removed in B6: ... the free `load`/`load_buffer`/`save` functions ... verified: zero `.load(` call sites workspace-wide." Fuel is lazy-only and has no eager write path to a model-file format anywhere in the tree today. There is nothing to migrate on the saving side because there is nothing there — the original plan's silence on saving is correct, not a gap. |

---

## Part B — `quantized/` split, more granular than "the whole directory moves"

The original plan's Part 3 treated `fuel-core/src/quantized/mod.rs` (and by
extension its sibling directory at `fuel-loaders/src/quantized/`) as one
row with one destination. Read individually:

| File | Content | Couples to `Tensor`/`Device`? | Proposed |
|---|---|---|---|
| `arch.rs`, 7,040 B | Architecture family detection from GGUF metadata/tensor-name patterns (`Llama`, `Qwen2`, `Phi3`, …) — pure classification over parsed metadata | No | MLMF |
| `gguf_file.rs`, 1,720 B | Re-exports `fuel_formats::gguf`'s wire types, wraps the header in a local `Content` struct | No (its own doc says B6 removed the tensor-construction half: `Content::tensor`, `tensor_from_mmap`, `write`) | MLMF |
| `gguf_mmap.rs`, 4,791 B | Not read in this amendment — needs its own check before scheduling; name suggests mmap-backed tensor *data* access, which may couple to `Device`/`Storage` unlike its siblings | Unverified | Flag — read before scheduling |
| `tokenizer.rs`, 12,071 B | Not read in this amendment — vocab/tokenizer parsing is plausibly "model-related file" loading (tokenizer.json ships in every HF model repo) but is a distinct artifact from the weight tensors themselves | Unverified | Flag — read before scheduling; likely MLMF but not asserted here |
| `config_from_gguf.rs`, 16,805 B | Not read in this amendment — largest file in the directory; "config from GGUF metadata" sits at the boundary between "parsing a model file" (MLMF) and "producing fuel's own model config structs" (fuel) | Unverified | Flag — needs a dedicated read; likely splits rather than moves wholesale, same shape as the original plan's `storage.rs`/`backend.rs` caveats |
| `imatrix_file.rs`, 720 B | Not read in this amendment — likely a thin wrapper around `fuel-formats::imatrix`, which per Part A stays in Fuel | Unverified | Almost certainly **stays in Fuel**, tied to the `imatrix.rs` carve-out — verify before asserting |

Two of six read fully, four flagged rather than guessed — same discipline
the original plan applied to `judge`/`planner`/`factories`/`utils`: a
complete-looking table that quietly assigns plausible homes to unread
files is worse than one that names the gaps.

---

## Part C — `fuel-formats` and `fuel-loaders` as crates, not just their contents

Once the moves above (and the flagged ones, once resolved) land, both
crates are reduced to:

- **`fuel-formats`**: only `imatrix.rs` (the carve-out) plus `lib.rs`.
  Whether a single-module crate is still worth keeping separate, or
  whether the raw imatrix parser folds into whatever fuel crate ends up
  owning quantization-adjacent code, is a real question this amendment
  does not answer — flagged for CireSnave/PM, not decided unilaterally.
- **`fuel-loaders`**: becomes a thin consumer of MLMF's API (parse via
  MLMF, then build fuel `Tensor`/`Storage` from the result) rather than a
  home for parsing logic itself. This mirrors the shim pattern the
  original plan already used for `fuel-core` → `fuel-loaders`: `fuel-core`
  kept `pub use` shims so callers didn't break; `fuel-loaders` doing the
  same for `mlmf` re-exports is the same shape one crate further down the
  chain.

Neither crate's own version bump or Cargo.toml dependency addition is
part of this amendment — no code changes here, per instruction.

---

## Part E — Reconciled against MLMF's published inventory (`ciresnave/mlmf#101`)

Read in full (384 lines) after this amendment's Parts A–C were drafted, to
check for disagreement rather than to source the amendment. None of the
per-file destinations above needed to change; three things are worth
recording because #101 either corroborates, sharpens, or exposes a gap
this document's grep-only method could not have found on its own.

- **Independent corroboration, not restatement.** #101 was produced in a
  separate session, against the mlmf repo, using its own method (reading
  mlmf's crate contents against fuel's `docs/gaps.md`), and reaches the
  same three calls this amendment reaches by a different route: the
  `imatrix.rs` raw-positional format stays Fuel's (#101 §1.4: "two
  different file formats that happen to share a name... should stay
  Fuel's for now" — same conclusion, same reasoning, no shared source
  beyond the item-63a ruling both documents cite); fuel's `arch.rs` is not
  a name-mapping module and nothing generic exists to relocate (#101 §5
  reaches this independently, by reading `arch.rs` directly rather than
  grepping for a name-mapping module that isn't there); saving has no
  live fuel-side implementation to migrate (#101 §2/§8 item 6 corroborates
  from the mlmf side — write capability exists only in mlmf's own legacy
  crate, scheduled for a rewrite, so there is nothing for fuel to adopt
  yet regardless). Per [`evidence-that-is-not-independent`](../method-rules.md#evidence-that-is-not-independent),
  agreement is only worth noting when the two artifacts were derived
  separately — these were, so it is recorded rather than assumed.
- **#101 is more granular where this document could only flag.** Part B's
  four unread `quantized/` files get partial answers from #101's capability
  tables: `gguf_file.rs`/`gguf_mmap.rs`'s content maps onto §1.1/§1.2's
  "container parse" and "mmap-backed zero-copy" rows (§1.2 explicitly
  names Fuel's mmap-drop-after-decode as GAP-204, separate from this
  amendment's flag); `tokenizer.rs` maps onto §4's `tokenizer.json`/
  `tokenizer_config.json` rows (mlmf **HAS** both, via `mlmf-meta`).
  `config_from_gguf.rs` is the one Part B flagged as sitting "right at the
  fuel/MLMF boundary" — #101 §4 resolves that boundary question directly:
  it reads `rope_theta` at `config_from_gguf.rs:210` and finds it already
  propagates absence via `?` rather than fabricating `10000.0` (the two
  `10000.0` literals nearby are test fixtures, not production fallbacks) —
  i.e. this file already meets the §6-fence discipline #101 holds mlmf
  itself to, which is a point in favor of migrating it rather than a
  reason to hold it back. `imatrix_file.rs` is not directly addressed by
  #101; still flagged, unread, in Part B.
- **A gap #101 does not cover: `fuel-formats/src/pickle.rs` (PyTorch
  `.pth`/pickle).** #101's format inventory (§1) covers GGUF, safetensors,
  ONNX, and imatrix — pickle does not appear anywhere in its 384 lines.
  This amendment's Part A still proposes MLMF as the destination (no
  backend-type reference, same transport-independent contract as the
  other three `fuel-formats` modules), but unlike `gguf.rs`/`safetensors.rs`
  that proposal now has **no corroborating capability entry on the mlmf
  side to point at** — mlmf may not read PyTorch checkpoints at all today.
  Flagged as a genuine open question for the mlmf lane, not asserted as
  covered.
- **#101's ordered migration plan (§8) supersedes any sequencing implied
  by this document.** This amendment does not propose an order — Part A is
  a per-file "does this move" table, not a schedule. Where the two need to
  be read together: #101 §8 item 1 (GGUF metadata + dtype table) has no
  fuel-side gating; item 5 (config.json/tokenizer readers) is gated on
  mlmf's own `#48` (`ModelConfig`'s `Option<T>` representation); items 6–7
  (saving, name-mapping) are gated on mlmf promoting `src/saver.rs`/
  `src/name_mapping.rs` out of its legacy crate first. Anyone scheduling
  work off this amendment should read #101 §8, not infer an order from
  Part A's table order (which is read-order, not priority-order).

---

## Part D — What this amendment does not do

- Does not execute any move. No code changed, no `Cargo.toml` touched, no
  version bump.
- Does not resolve the four flagged-unread files in Part B
  (`gguf_mmap.rs`, `tokenizer.rs`, `config_from_gguf.rs`,
  `imatrix_file.rs` — three of four now partially addressed by #101, see
  Part E), or the `model_progress.rs` weight-dependency question (#101
  does not cover progress reporting either), or the `fuel-formats`-as-a-crate
  question in Part C.
- Does not invent MLMF-side module names. Every destination above says
  "MLMF," not `mlmf_gguf::...` or similar — #101 §7 is the place that
  names exact target modules/crates; this document points at it (Part E)
  rather than restating it.
- Does not touch the original plan's Parts 2 and 4 (the `Device`/`Tensor`
  orphan-rule cycle and the `Tensor`-destination decision) — neither is a
  model-file-loading concern, and CireSnave's mlmf rule does not bear on
  either.
- Does not reopen item 63a's imatrix split — that ruling stands and this
  amendment treats it as a constraint, not a candidate for
  reconsideration.
