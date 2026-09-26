# fuel-core: a complete per-module destination plan

**Status: PLAN ONLY, not executed.** No slice in this document has been started. Written per CireSnave's directive ("if we don't have a plan for where everything from fuel-core moves to, lets make one now") and his ruling that fuel-core is to be GONE, not renamed.

**Method:** every module is enumerated from `git ls-tree -r --name-only origin/main -- fuel-core/src/`, not from `lib.rs`'s `pub mod` list, so a private module still gets a row. 38 files/dirs total. Where `docs/architecture/02-layers.md` already pins a destination, this doc uses it and says so — it does not re-decide it. Where this plan disagrees, that is called out explicitly as a proposed change, not a quiet reroute.

**Consumer counts:** four are the ledger's own compiler-census numbers, used as given (`safetensors` 219, `hf_config` 34, `quantized` ~24, `model_progress` 0). The rest are a same-methodology proxy — `git grep` for `fuel_core::<module>`/`fuel::<module>` path references outside `fuel-core/` itself — which is NOT a full compiler census (it misses re-exported-then-renamed access and counts a file once regardless of use-site count), but is consistent across every row and far better than nothing. Rows needing a real census before scheduling are marked.

---

## Part 1 — Pure re-export shims (real content already lives elsewhere)

These eight files contain no original logic. Verified by reading each file directly, not inferred from size. The correct move is **delete the shim + repoint every consumer to the real crate** — there is no "content to migrate," only paths to fix.

| Module | Real home (already exists) | Confirmed by |
|---|---|---|
| `dtype.rs` | `fuel_ir::dtype` | file's own doc: "defined in fuel-core-types [now fuel-ir] and re-exported here" |
| `shape.rs` | `fuel_ir::shape` | file's own doc: "re-exports... adds nothing of its own" |
| `storage.rs` | `fuel_ir::storage` (+ `StorageApplyOps` ext trait, needs a decision — see below) | file's own doc: "the Storage struct... moved to fuel-core-types [fuel-ir]" |
| `strided_index.rs` | `fuel_ir::strided_index` | `pub use fuel_ir::strided_index::*;` — the entire file body |
| `layout.rs` | `fuel_ir::layout` | `pub use fuel_ir::layout::*;` — the entire file body |
| `error.rs` | `fuel_ir::error` | `pub use fuel_ir::error::{Context, Error, ...};` — the entire file body |
| `dyn_backend.rs` | `fuel_backend_contract::dyn_backend` | `pub use fuel_backend_contract::dyn_backend::{...};` — the entire file body |
| `cpu_backend/mod.rs` | `fuel_ir::{CpuDevice, CpuStorage, ...}` | file's own doc: "all real CPU kernel logic lives in fuel-cpu-backend" |

**`storage.rs` caveat:** it is *almost* a pure shim, but also defines a local `StorageApplyOps` trait extension (per its own doc, "Phase 7.5 work item G fix-up"). That trait needs its own destination decision — most likely `fuel-memory` alongside `Storage` itself (02-layers.md: "fuel-memory: the in-memory storage substrate only — the Storage struct, the BackendStorage enum, and allocators"), but this plan does not assert that without reading the trait's actual shape first. Flagged, not resolved.

**`backend.rs`** is adjacent but not identical: it still surfaces one real trait, `HostStorage` (a capability marker), described as "the only trait still surfaced through this module" after an eager era of larger backend traits was deleted. Needs its own read before a destination call — likely `fuel-backend-contract` alongside `DynBackendDevice`/`DynBackendStorage`, but not asserted here without checking `HostStorage`'s actual definition and whether `fuel-backend-contract` already has an equivalent.

**Consumer counts for this group:** not yet measured by compiler census for any of these eight — they need one before scheduling, same discipline the ledger applied to `safetensors`/`hf_config`. Given they're pure re-exports, the repoint should be pure mechanical `sed`, same shape as the `safetensors` Slice-3(A) batches already running.

---

## Part 2 — The three GPU-backend bridge files: a genuine cycle hazard, not a simple move

| Module | Size | What it actually contains |
|---|---|---|
| `cuda_backend/mod.rs` | 2,389 B | `impl From<CudaDevice> for Device` + free fns needing `crate::Device` in their signature |
| `metal_backend/mod.rs` | 2,017 B | same shape, `From<MetalDevice> for Device` |
| `vulkan_backend.rs` | 2,240 B | same shape, `From<VulkanBackend> for Device` |

Unlike `cpu_backend/mod.rs`, these are **not** shims — each owns a real trait impl that the Rust orphan rule pins to whichever crate owns one of the two types in the `impl From<Foreign> for Local`. `CudaDevice`/`MetalDevice`/`VulkanBackend` are foreign (they live in `fuel-cuda-backend`/`fuel-metal-backend`/`fuel-vulkan-backend`); `Device` is the local type these impls need, and `Device` is defined in `fuel-core/src/device.rs` itself — not a shim, a real 14,880-byte struct.

**This means these three files cannot be assigned a destination independently of `Device`'s destination.** Whichever crate ends up owning `Device` is where all three orphan-rule impls must live (or each backend crate could define the reverse `impl From<Device> for X`, but that changes call-site ergonomics and is a real design choice, not a mechanical move). **This is not a separate hard case — it is the Tensor/Device question (Part 4) wearing three different filenames.**

---

## Part 3 — The rest of the module table

| Module | Destination (pinned / proposed) | Crate exists? | Consumers (method) | Cycle hazard |
|---|---|---|---|---|
| `hf_config.rs` | `fuel-loaders` | yes | **34** (ledger compiler census) | none — already partly landed pattern (Slice 2) |
| `model_progress.rs` | `fuel-loaders` | yes | **0** (ledger census) — delete outright, no repoint needed | none |
| `quantized/mod.rs` | `fuel-loaders::quantized` | yes | **~24** (ledger census, consumers are `fuel-transformers` quantized loaders) | none — `fuel-loaders/src/quantized/` already exists (arch.rs, config_from_gguf.rs, gguf_file.rs, gguf_mmap.rs, imatrix_file.rs, tokenizer.rs); this file is thin next to its sibling directory already there |
| `safetensors.rs` | `fuel-loaders::safetensors` | **already executed** (Slice 3) | **219** genuine, 222 total mentions (this session's census, batch 1 of 219 landed as #259) | none — shim already lands clean |
| `device.rs` | **UNSCOPED — see Part 4** | n/a | not separately measured; couples everything that touches `Device` | **yes — the central one** |
| `lazy.rs` | **UNSCOPED — see Part 4** | n/a | effectively all of fuel-core's own other modules, plus `fuel-nn`, `fuel-transformers` | **yes — the central one** |
| `cuda_backend/mod.rs` | tied to `device.rs` (Part 2) | — | not separately measured | yes |
| `metal_backend/mod.rs` | tied to `device.rs` (Part 2) | — | not separately measured | yes |
| `vulkan_backend.rs` | tied to `device.rs` (Part 2) | — | not separately measured | yes |
| `judge/{mod,cache,oracle}.rs` | proposed: `fuel-dispatch` (it IS the ranker/selector's runtime-measurement half; 02-layers.md assigns "the ranker/selector chain" to fuel-dispatch) | yes | 7 (path-grep proxy) | needs check: does `judge` construct `Tensor`/`Device` directly for its benchmarking harness? Not verified here — likely yes, given it profiles real kernels, which would make it depend on wherever `Tensor` lands |
| `planner.rs` | **GENUINELY UNSCOPED** | — | 0 (path-grep proxy) | not checked — 4,515 B, small enough to read before deciding, not done here |
| `train.rs` | proposed: new crate `fuel-training`† (02-layers.md names training as a Models-layer or above concern; no existing crate owns SGD/AdamW/loss training loops) | **no — would need creating** | 3 (path-grep proxy) | almost certainly needs `Tensor` for gradient computation — coupled to Part 4 |
| `kv_block_pool.rs` | proposed: `fuel-inference` (KV-cache/serving concern; `fuel-inference::multi_session::DecodeModel` already lives there per 02-layers.md's Models-layer section) | yes | 3 (path-grep proxy) | needs check |
| `kv_block_pool_device.rs` | same as above, `fuel-inference` | yes | 3 (path-grep proxy) | needs check — name suggests device-residency, likely couples to `Device` |
| `inference_context.rs` | proposed: `fuel-inference` | yes | 21 (path-grep proxy) | almost certainly couples to `Tensor` — this is the multi-session/KV context object |
| `persistent_decode.rs` | proposed: `fuel-inference` | yes | 9 (path-grep proxy) | couples to `Tensor`/`Device` by name |
| `pipelined_bridge.rs` | **GENUINELY UNSCOPED** | — | 14 (path-grep proxy) | 127,224 B — one of the largest files in the crate; this session's own earlier work today (the sub-byte-dtype investigation, GAP-339) read deep into this file and found it entangled with the executor/dispatch boundary. Needs its own dedicated read, not a plausible-sounding assignment |
| `decode_shape.rs` | proposed: alongside wherever `lazy.rs`/`Tensor` lands (decode-plan shape geometry, directly serves `Tensor::forward_paged_step` family) | — | 30 (path-grep proxy) | tied to Part 4 |
| `decode_state_spec.rs` | proposed: `fuel-inference`, alongside `kv_block_pool*`/`inference_context` | yes | 1 (path-grep proxy) | needs check |
| `lazy_latent_cache.rs` | tied to `lazy.rs` (Part 4) — name says it directly | — | 3 (path-grep proxy) | yes |
| `scheduling.rs` | proposed: `fuel-dispatch` (02-layers.md: "the ranker/selector chain... PipelinedExecutor" is fuel-dispatch's job; this file's `dtype_size_bytes`/backend-placement logic reads as the same family) | yes | 3 (path-grep proxy) | needs check — this session read parts of this file earlier tonight (the sub-byte dtype size table) and it did not appear to construct `Tensor` directly, but that was a narrow read, not exhaustive |
| `nf4.rs` | proposed: `fuel-quantized` (NF4 is a quantization scheme; 02-layers.md places `fuel-quantized` as a Foundation-layer crate with "substantial standalone value") | yes | 0 (path-grep proxy) — but this is misleading, see note | needs check — `nf4.rs`'s own doc says it is "companion to `fuel_graph::NodeHandle::nf4_matmul`", so it likely couples to `Tensor`'s `nf4_matmul` method even though nothing imports the *module* by path (the coupling would show as a `Tensor` method call site, not a `fuel_core::nf4::` import — the path-grep proxy is blind to this shape of consumer, flagged explicitly rather than trusted) |
| `telemetry.rs` | proposed: `fuel-dispatch::telemetry` (that module already exists and is the far larger, real telemetry surface — this file may be a thin forwarding shim, not verified) | yes | 1 (path-grep proxy) | not checked — needs a content read before any claim |
| `factories.rs` | **GENUINELY UNSCOPED** | — | 0 (path-grep proxy) | 14,002 B, doc says "the Judge's realize seam" — coupled to `judge` and therefore to whatever `judge` depends on |
| `accelerate.rs` | proposed: `fuel-cpu-backend` (Apple Accelerate is a CPU BLAS backend; parallel structure to `mkl.rs`) | yes | not measured | needs check |
| `mkl.rs` | proposed: `fuel-cpu-backend` or `fuel-mkl-cpu-backend` (which already exists per CLAUDE.md's own build-discipline notes) | yes (`fuel-mkl-cpu-backend`) | not measured | needs check |
| `utils.rs` | **GENUINELY UNSCOPED** — 2,593 B, generic name, needs a content read to even guess | — | not measured | unknown |
| `test_utils.rs` | stays with whatever `lazy.rs`/`Tensor`'s test infrastructure needs it (likely follows `Tensor`, not a separate decision) | — | not measured | tied to Part 4 |
| `lib.rs` | dissolves entirely once every module above has a home — this is the file that currently re-exports everything; it does not move anywhere, it disappears | n/a | n/a | n/a — this IS the "fuel-core is GONE" condition being satisfied |

---

## Part 4 — Where does `Tensor` go? (the question a plan cannot skip)

**`Tensor` is defined once, in `fuel-core/src/lazy.rs`, 458,752 bytes — the largest file in the crate by roughly 2.7x over the next largest (`judge/mod.rs` at 172,141 B).** The dissolution doc's own framing (quoted in the task, not invented here) calls it "NOT a mechanically-splittable monolith," and this plan does not contradict that by pretending otherwise.

**The honest answer: `Tensor`'s home is a decision CireSnave has not made, and no amount of module-table bookkeeping substitutes for it.** Naming the options rather than picking one:

1. **A new dedicated crate** (`fuel-tensor`†, already named with the `†` unbuilt-but-ratified marker in 02-layers.md: *"fuel-tensor† + fuel-autograd† (post-fission per Phase 7.5 work item E): the user-facing handle + autograd story. fuel-tensor† wraps fuel-graph"*). **This destination is already pinned in the architecture doc** — it is not a new proposal, it is a ratified-but-unbuilt name this plan is pointing at. If this is the answer, `lazy.rs`'s split is the real Slice, and everything in Part 2/3 marked "tied to Part 4" resolves once `fuel-tensor`† exists: the three GPU-backend bridge files move there, `decode_shape.rs`/`lazy_latent_cache.rs`/`test_utils.rs` follow.
2. **Split `Tensor` itself** into a thin handle (which could live in `fuel-tensor`†) plus the "eager-dispatch convenience methods" that 02-layers.md's `storage.rs` note says were *already* partly extracted once ("almost all of its eager-dispatch methods moved to fuel-core-types::storage") — meaning there is *some* precedent inside this very file for peeling layers off `Tensor` rather than moving it as one block. Whether that precedent generalizes to the rest of `lazy.rs`'s 458KB is unverified.
3. **Leave `Tensor` where it lands last** — i.e., treat `lazy.rs` as the true final remnant of `fuel-core`, dissolve everything else first, and let what's left (which may just be `Tensor` + `Device` + the three backend bridges) become the crate that used to be called `fuel-core` under a new, non-colliding name. This is explicitly **not** what CireSnave ruled ("GONE, not renamed"), so this option is named only to be ruled out, not proposed.

**This plan's recommendation, stated as a recommendation and not a decision:** option 1 (the already-`†`-marked `fuel-tensor` destination) is the one that requires inventing nothing — the name and its scope description already exist in `02-layers.md`. The open work is building it, splitting `lazy.rs` into it, and repointing `Device`'s three backend-bridge impls alongside it. This plan treats that as **the** hard slice — not a row to schedule casually alongside `hf_config`'s 34-consumer repoint.

---

## Part 5 — Sequencing

Ordered cycle-free-first, then ascending consumer count, per instruction. Shim-debt ledger's four outstanding repoints folded in at their measured position.

| Order | Item | Consumers | Why here |
|---|---|---|---|
| 1 | `model_progress.rs` | 0 | Ledger's own call: delete the shim outright, easiest item, zero repoint |
| 2 | `factories.rs`, `planner.rs`, `utils.rs`, `telemetry.rs` | 0-1 each (proxy) | Need a content read each before real scheduling, but low consumer count means low blast radius once read |
| 3 | The 8 pure-shim files (Part 1) | not yet measured | Mechanical repoint once censused — same shape as `safetensors`, should be cheap per file, but 8 files × unknown-but-likely-large consumer counts (these are core vocabulary: `DType`/`Shape`/`Layout` are probably imported almost everywhere) means this could be the single largest mechanical sweep in the whole plan. **Census before scheduling, per the same rule the PM applied to `safetensors`.** |
| 4 | `quantized/mod.rs` | ~24 | Ledger census, low enough to batch with fuel-loaders' existing `quantized/` sibling directory |
| 5 | `hf_config.rs` | 34 | Ledger census |
| 6 | `nf4.rs`, `accelerate.rs`, `mkl.rs` | not measured, likely low by path-grep but `nf4.rs` specifically flagged as undercounted by this method | Read each file first; `nf4.rs`'s real coupling is to `Tensor::nf4_matmul`, not a module-path import, so its true consumer count needs a `Tensor` method-call census, not the path-grep proxy this table otherwise uses |
| 7 | `safetensors.rs` consumer repoint (batches 2+) | 219 total, 1/219 (fuel-nn) landed as #259, paused pending gate | Already in flight, largest known-mechanical sweep |
| 8 | `judge/*`, `scheduling.rs` | 7, 3 (proxy) | Proposed fuel-dispatch destination per 02-layers.md's existing pin for the ranker/selector chain, but needs the `Tensor`-coupling check noted in Part 3 before treating as purely mechanical |
| 9 | `kv_block_pool.rs`, `kv_block_pool_device.rs`, `inference_context.rs`, `persistent_decode.rs`, `decode_state_spec.rs` | 3-21 (proxy) | Proposed `fuel-inference` destination; all five almost certainly couple to `Tensor`/`Device`, so this batch cannot start before Part 4 is resolved |
| 10 | `pipelined_bridge.rs` | 14 (proxy) | Explicitly named GENUINELY UNSCOPED — needs its own dedicated read (this session's GAP-339 work touched it but did not scope a destination) |
| 11 | **`lazy.rs` + `device.rs` + the 3 GPU-backend bridges + `decode_shape.rs` + `lazy_latent_cache.rs` + `test_utils.rs` + `train.rs`** | effectively-all | **This is Part 4. Cannot be scheduled as a normal batch — it needs CireSnave's ruling on `Tensor`'s destination first**, per the recommendation above |
| 12 | `lib.rs` | n/a | Deleted last, once nothing remains to re-export — this is the literal moment `fuel-core` stops existing |

---

## Part 6 — Genuinely unscoped (named here so the table has no hidden gaps)

Per instruction: a complete-looking table that quietly assigns plausible homes to hard cases is worse than one that names them. These five have **no destination proposed in this document**, only a flag that they need a dedicated read before anyone proposes one:

- **`planner.rs`** (4,515 B) — 0 measured consumers by the path-grep proxy, but that proxy has already been shown (via `nf4.rs`) to miss method-call-shaped coupling. Unread.
- **`pipelined_bridge.rs`** (127,224 B) — second-largest file in the crate; touched but not scoped by this session's earlier GAP-339 investigation.
- **`utils.rs`** (2,593 B) — generic name, unread.
- **`storage.rs`'s `StorageApplyOps` trait** — not the whole file (which is a shim per Part 1), just this one extension trait's real destination.
- **`backend.rs`'s `HostStorage` trait** — same shape as above, one real trait inside an otherwise-thin file.

And the largest genuinely unscoped item of all, treated separately because it is not a module but a decision: **`Tensor`'s crate**, Part 4.

---

## What this plan does not do

It does not execute any slice. It does not start Slice 3 batch 2. It does not invent a destination where `02-layers.md` is silent — five items stay named-and-unscoped rather than assigned. It does not treat the `Tensor` question as answerable by table-filling. It proposes `fuel-training`† as a new crate name for `train.rs` since none of the existing crates own that responsibility, and flags that proposal as a proposed *change* (a new crate to create) rather than a pinned destination, per instruction to say so explicitly when disagreeing with or extending the existing architecture doc.
